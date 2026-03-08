-- Database Initialization Script
-- Creates initial database structure and required extensions

-- Enable required PostgreSQL extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For text search

-- Create application user if not exists (for Docker initialization)
DO
$$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'wiki_user') THEN
        CREATE ROLE wiki_user WITH LOGIN PASSWORD 'wiki_password';
    END IF;
END
$$;

-- Grant necessary privileges
GRANT ALL PRIVILEGES ON DATABASE wikipedia_intelligence TO wiki_user;

-- Create schema for application tables
CREATE SCHEMA IF NOT EXISTS wiki_intel;
GRANT ALL ON SCHEMA wiki_intel TO wiki_user;

-- Set default schema
ALTER DATABASE wikipedia_intelligence SET search_path TO wiki_intel, public;

-- Create audit log table
CREATE TABLE IF NOT EXISTS wiki_intel.audit_log (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    event_type VARCHAR(50) NOT NULL,
    table_name VARCHAR(100),
    record_id BIGINT,
    user_name VARCHAR(100),
    changes JSONB,
    ip_address INET
);

CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp ON wiki_intel.audit_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_log_event_type ON wiki_intel.audit_log(event_type);

-- Create system health table
CREATE TABLE IF NOT EXISTS wiki_intel.system_health (
    id SERIAL PRIMARY KEY,
    check_timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    component VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL,
    message TEXT,
    metrics JSONB
);

CREATE INDEX IF NOT EXISTS idx_system_health_timestamp ON wiki_intel.system_health(check_timestamp);
CREATE INDEX IF NOT EXISTS idx_system_health_component ON wiki_intel.system_health(component);

-- Create configuration table for runtime config
CREATE TABLE IF NOT EXISTS wiki_intel.runtime_config (
    key VARCHAR(100) PRIMARY KEY,
    value TEXT NOT NULL,
    value_type VARCHAR(20) NOT NULL,
    description TEXT,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_by VARCHAR(100)
);

-- Insert default runtime configuration
INSERT INTO wiki_intel.runtime_config (key, value, value_type, description)
VALUES 
    ('dashboard_refresh_interval', '300', 'integer', 'Dashboard auto-refresh interval in seconds'),
    ('cache_ttl', '300', 'integer', 'Cache TTL in seconds'),
    ('hype_threshold', '0.75', 'float', 'Threshold for flagging trending articles'),
    ('reputation_alert_threshold', '0.7', 'float', 'Threshold for reputation risk alerts')
ON CONFLICT (key) DO NOTHING;

-- Create function to update timestamp
CREATE OR REPLACE FUNCTION wiki_intel.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Success message
DO $$
BEGIN
    RAISE NOTICE 'Database initialization completed successfully';
END $$;
