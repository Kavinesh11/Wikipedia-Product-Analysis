#!/usr/bin/env python3
"""
Health Check Script
Performs comprehensive health checks on all system components
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml
from sqlalchemy import create_engine, text
import redis
import requests


class HealthChecker:
    """Performs health checks on system components"""
    
    def __init__(self, config: dict):
        self.config = config
        self.results = {}
    
    def resolve_env_var(self, value):
        """Resolve environment variable references"""
        if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
            env_var = value[2:-1]
            return os.getenv(env_var, value)
        return value
    
    def check_database(self) -> Tuple[bool, str, dict]:
        """Check database health"""
        try:
            if self.config['database'].get('use_sqlite', False):
                db_path = self.config['database']['sqlite_path']
                db_url = f"sqlite:///{db_path}"
            else:
                host = self.resolve_env_var(self.config['database']['postgres_host'])
                port = self.config['database']['postgres_port']
                db = self.resolve_env_var(self.config['database']['postgres_db'])
                user = self.resolve_env_var(self.config['database']['postgres_user'])
                password = self.resolve_env_var(self.config['database']['postgres_password'])
                
                db_url = f"postgresql://{user}:{password}@{host}:{port}/{db}"
            
            engine = create_engine(db_url, pool_pre_ping=True)
            
            # Test connection
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchone()
                
                # Get database size
                if 'postgresql' in db_url:
                    size_result = conn.execute(text(
                        "SELECT pg_size_pretty(pg_database_size(current_database()))"
                    ))
                    db_size = size_result.fetchone()[0]
                else:
                    db_size = "N/A"
                
                # Get connection count
                if 'postgresql' in db_url:
                    conn_result = conn.execute(text(
                        "SELECT count(*) FROM pg_stat_activity"
                    ))
                    conn_count = conn_result.fetchone()[0]
                else:
                    conn_count = "N/A"
            
            metrics = {
                'database_size': db_size,
                'active_connections': conn_count,
                'type': 'PostgreSQL' if 'postgresql' in db_url else 'SQLite'
            }
            
            return True, "Database is healthy", metrics
            
        except Exception as e:
            return False, f"Database check failed: {str(e)}", {}
    
    def check_redis(self) -> Tuple[bool, str, dict]:
        """Check Redis health"""
        try:
            host = self.resolve_env_var(self.config['cache']['redis_host'])
            port = self.config['cache']['redis_port']
            db = self.config['cache']['redis_db']
            
            client = redis.Redis(
                host=host,
                port=port,
                db=db,
                socket_timeout=5,
                socket_connect_timeout=5
            )
            
            # Test connection
            client.ping()
            
            # Get Redis info
            info = client.info()
            
            metrics = {
                'version': info.get('redis_version', 'unknown'),
                'used_memory': info.get('used_memory_human', 'unknown'),
                'connected_clients': info.get('connected_clients', 0),
                'uptime_days': info.get('uptime_in_days', 0)
            }
            
            return True, "Redis is healthy", metrics
            
        except Exception as e:
            return False, f"Redis check failed: {str(e)}", {}
    
    def check_api(self) -> Tuple[bool, str, dict]:
        """Check Wikimedia API availability"""
        try:
            base_url = self.config['api']['wikimedia_base_url']
            timeout = self.config['api'].get('timeout', 30)
            
            # Test API endpoint
            start_time = datetime.now()
            response = requests.get(
                f"{base_url}/metrics/pageviews/",
                timeout=timeout
            )
            response_time = (datetime.now() - start_time).total_seconds()
            
            metrics = {
                'status_code': response.status_code,
                'response_time_seconds': round(response_time, 3),
                'endpoint': base_url
            }
            
            if response.status_code in [200, 404]:
                return True, "API is accessible", metrics
            else:
                return False, f"API returned status code: {response.status_code}", metrics
                
        except Exception as e:
            return False, f"API check failed: {str(e)}", {}
    
    def check_disk_space(self) -> Tuple[bool, str, dict]:
        """Check disk space"""
        try:
            import shutil
            
            # Check main directories
            directories = ['logs', 'data', 'output', 'backups']
            disk_info = {}
            
            for directory in directories:
                path = Path(__file__).parent.parent / directory
                if path.exists():
                    usage = shutil.disk_usage(path)
                    disk_info[directory] = {
                        'total_gb': round(usage.total / (1024**3), 2),
                        'used_gb': round(usage.used / (1024**3), 2),
                        'free_gb': round(usage.free / (1024**3), 2),
                        'percent_used': round((usage.used / usage.total) * 100, 2)
                    }
            
            # Check if any directory is over 90% full
            critical = any(info['percent_used'] > 90 for info in disk_info.values())
            
            if critical:
                return False, "Disk space critical (>90% used)", disk_info
            else:
                return True, "Disk space is adequate", disk_info
                
        except Exception as e:
            return False, f"Disk space check failed: {str(e)}", {}
    
    def check_log_files(self) -> Tuple[bool, str, dict]:
        """Check log files"""
        try:
            log_dir = Path(__file__).parent.parent / 'logs'
            
            if not log_dir.exists():
                return True, "No log files yet", {}
            
            log_files = list(log_dir.glob('*.log'))
            
            log_info = {}
            for log_file in log_files:
                size_mb = log_file.stat().st_size / (1024**2)
                log_info[log_file.name] = {
                    'size_mb': round(size_mb, 2),
                    'modified': datetime.fromtimestamp(log_file.stat().st_mtime).isoformat()
                }
            
            # Check if any log file is over 100MB
            large_logs = [name for name, info in log_info.items() if info['size_mb'] > 100]
            
            if large_logs:
                return False, f"Large log files detected: {', '.join(large_logs)}", log_info
            else:
                return True, "Log files are normal size", log_info
                
        except Exception as e:
            return False, f"Log file check failed: {str(e)}", {}
    
    def run_all_checks(self) -> Dict:
        """Run all health checks"""
        checks = {
            'database': self.check_database,
            'redis': self.check_redis,
            'api': self.check_api,
            'disk_space': self.check_disk_space,
            'log_files': self.check_log_files
        }
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'checks': {}
        }
        
        for check_name, check_func in checks.items():
            success, message, metrics = check_func()
            
            results['checks'][check_name] = {
                'status': 'healthy' if success else 'unhealthy',
                'message': message,
                'metrics': metrics
            }
            
            if not success:
                results['overall_status'] = 'unhealthy'
        
        return results


def load_config(environment: str = None) -> dict:
    """Load configuration"""
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    env = environment or os.getenv('ENVIRONMENT', 'development')
    
    if env not in config:
        raise ValueError(f"Environment '{env}' not found in config")
    
    return config[env]


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run health checks')
    parser.add_argument(
        '--environment',
        choices=['development', 'staging', 'production'],
        help='Environment to check'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON'
    )
    parser.add_argument(
        '--check',
        choices=['database', 'redis', 'api', 'disk_space', 'log_files'],
        help='Run specific check only'
    )
    
    args = parser.parse_args()
    
    try:
        # Load config
        config = load_config(args.environment)
        
        # Create health checker
        checker = HealthChecker(config)
        
        # Run checks
        if args.check:
            # Run specific check
            check_func = getattr(checker, f'check_{args.check}')
            success, message, metrics = check_func()
            
            if args.json:
                result = {
                    'status': 'healthy' if success else 'unhealthy',
                    'message': message,
                    'metrics': metrics
                }
                print(json.dumps(result, indent=2))
            else:
                status = "✓" if success else "✗"
                print(f"{status} {args.check}: {message}")
                if metrics:
                    print(f"  Metrics: {json.dumps(metrics, indent=2)}")
            
            sys.exit(0 if success else 1)
        else:
            # Run all checks
            results = checker.run_all_checks()
            
            if args.json:
                print(json.dumps(results, indent=2))
            else:
                print("\n" + "="*60)
                print("Health Check Results")
                print("="*60)
                print(f"Timestamp: {results['timestamp']}")
                print(f"Overall Status: {results['overall_status'].upper()}")
                print()
                
                for check_name, check_result in results['checks'].items():
                    status = "✓" if check_result['status'] == 'healthy' else "✗"
                    print(f"{status} {check_name}: {check_result['message']}")
                    
                    if check_result['metrics']:
                        for key, value in check_result['metrics'].items():
                            print(f"    {key}: {value}")
                    print()
            
            # Exit with error code if unhealthy
            sys.exit(0 if results['overall_status'] == 'healthy' else 1)
            
    except Exception as e:
        print(f"Health check failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
