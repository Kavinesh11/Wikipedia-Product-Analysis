# Architecture Documentation

## System Overview

The Wikipedia Intelligence System is a production-ready business intelligence platform that transforms Wikipedia data into actionable business insights. The system follows a layered architecture with five primary components operating in a data pipeline pattern.

## High-Level Architecture

```mermaid
graph TB
    subgraph "External Data Sources"
        API[Wikimedia Pageviews API]
        EDIT[Wikipedia Edit History API]
        WIKI[Wikipedia Articles]
    end
    
    subgraph "Data Ingestion Layer"
        PVC[Pageviews Collector]
        EHS[Edit History Scraper]
        CRAWLER[Crawl4AI Pipeline]
        BTF[Bot Traffic Filter]
        RL[Rate Limiter]
    end
    
    subgraph "Processing Layer"
        ETL[ETL Pipeline Manager]
        CLEAN[Data Cleaning]
        VALID[Validation]
        AGG[Aggregations]
        FEAT[Feature Engineering]
        DQ[Data Quality Monitor]
    end
    
    subgraph "Storage Layer"
        PG[(PostgreSQL<br/>Data Warehouse)]
        REDIS[(Redis Cache)]
        DW[Dimensional Model]
    end
    
    subgraph "Analytics Layer"
        TS[Time Series<br/>Forecaster]
        CLUSTER[Topic<br/>Clustering]
        HYPE[Hype<br/>Detection]
        REP[Reputation<br/>Monitor]
        KG[Knowledge<br/>Graph]
    end
    
    subgraph "Visualization Layer"
        DASH[Dashboard App]
        ALERTS[Alert System]
        EXPORT[Export Service]
    end
    
    subgraph "Orchestration"
        SCHED[Job Scheduler]
        HEALTH[Health Monitor]
        LOGS[Logging System]
    end
    
    API --> PVC
    EDIT --> EHS
    WIKI --> CRAWLER
    
    PVC --> RL
    EHS --> RL
    CRAWLER --> RL
    
    RL --> BTF
    BTF --> ETL
    
    ETL --> CLEAN
    CLEAN --> VALID
    VALID --> AGG
    AGG --> FEAT
    FEAT --> DQ
    
    DQ --> PG
    DQ --> REDIS
    PG --> DW
    
    DW --> TS
    DW --> CLUSTER
    DW --> HYPE
    DW --> REP
    DW --> KG
    
    TS --> DASH
    CLUSTER --> DASH
    HYPE --> DASH
    REP --> ALERTS
    KG --> DASH
    
    DASH --> EXPORT
    
    SCHED --> PVC
    SCHED --> EHS
    SCHED --> CRAWLER
    SCHED --> TS
    
    HEALTH --> LOGS
    ETL --> LOGS
    ALERTS --> LOGS
```

## Component Details

### Data Ingestion Layer

The Data Ingestion Layer collects data from three primary sources using specialized collectors.

#### Components

**Pageviews Collector**
- Queries Wikimedia Pageviews API
- Filters bot traffic automatically
- Segments by device type (desktop, mobile-web, mobile-app)
- Implements exponential backoff for rate limits
- Validates response schemas

**Edit History Scraper**
- Extracts revision data from Wikipedia API
- Classifies editors (anonymous vs registered)
- Detects reverted edits (vandalism signals)
- Calculates edit velocity metrics
- Tracks rolling window statistics

**Crawl4AI Pipeline**
- Performs asynchronous web crawling
- Extracts structured content (infoboxes, tables, categories)
- Implements BFS for deep crawls
- Respects robots.txt and rate limits
- Supports checkpoint/resume for long operations

**Supporting Components**
- **Rate Limiter**: Token bucket algorithm, 200 req/sec limit
- **Bot Traffic Filter**: Separates human from automated traffic
- **API Client**: Connection pooling, retry logic, circuit breakers

#### Data Flow

```mermaid
sequenceDiagram
    participant API as Wikimedia API
    participant RL as Rate Limiter
    participant PVC as Pageviews Collector
    participant ETL as ETL Pipeline
    
    PVC->>RL: Request token
    RL-->>PVC: Token granted
    PVC->>API: Fetch pageviews
    API-->>PVC: Response (JSON)
    PVC->>PVC: Validate schema
    PVC->>PVC: Filter bots
    PVC->>ETL: Send PageviewRecords
```

### Processing Layer

The Processing Layer transforms raw data into analytics-ready formats through ETL pipelines.

#### Components

**ETL Pipeline Manager**
- Orchestrates data transformation workflows
- Implements validation rules
- Performs deduplication
- Tracks data lineage
- Monitors pipeline health

**Data Cleaning**
- Handles missing values
- Removes outliers
- Normalizes formats
- Fixes encoding issues

**Validation**
- Schema validation
- Business rule checks
- Referential integrity
- Quarantines invalid records

**Aggregations**
- Hourly, daily, weekly, monthly rollups
- Platform aggregations
- Cluster-level metrics
- Trend calculations

**Feature Engineering**
- Growth rate calculations
- Velocity metrics
- Density scores
- Composite indicators

#### Pipeline Architecture

```mermaid
graph LR
    subgraph "ETL Pipeline"
        INPUT[Raw Data]
        VALIDATE[Validate]
        CLEAN[Clean]
        TRANSFORM[Transform]
        DEDUPE[Deduplicate]
        LOAD[Load]
        QUARANTINE[Quarantine]
    end
    
    INPUT --> VALIDATE
    VALIDATE -->|Valid| CLEAN
    VALIDATE -->|Invalid| QUARANTINE
    CLEAN --> TRANSFORM
    TRANSFORM --> DEDUPE
    DEDUPE --> LOAD
    LOAD --> OUTPUT[Data Warehouse]
```

### Storage Layer

The Storage Layer provides persistent storage and caching for analytics data.

#### Database Schema (Star Schema)

```mermaid
erDiagram
    FACT_PAGEVIEWS ||--o{ DIM_ARTICLES : "article_id"
    FACT_PAGEVIEWS ||--o{ DIM_DATES : "date_id"
    FACT_EDITS ||--o{ DIM_ARTICLES : "article_id"
    FACT_CRAWL_RESULTS ||--o{ DIM_ARTICLES : "article_id"
    MAP_ARTICLE_CLUSTERS ||--o{ DIM_ARTICLES : "article_id"
    MAP_ARTICLE_CLUSTERS ||--o{ DIM_CLUSTERS : "cluster_id"
    AGG_ARTICLE_METRICS ||--o{ DIM_ARTICLES : "article_id"
    AGG_CLUSTER_METRICS ||--o{ DIM_CLUSTERS : "cluster_id"
    
    FACT_PAGEVIEWS {
        bigint id PK
        int article_id FK
        int date_id FK
        int hour
        varchar device_type
        int views_human
        int views_bot
        int views_total
    }
    
    FACT_EDITS {
        bigint id PK
        int article_id FK
        bigint revision_id
        timestamp timestamp
        varchar editor_type
        boolean is_reverted
        int bytes_changed
    }
    
    DIM_ARTICLES {
        int id PK
        varchar title
        text url
        varchar namespace
        timestamp first_seen
        timestamp last_updated
    }
    
    DIM_DATES {
        int id PK
        date date
        int year
        int quarter
        int month
        int week
        int day_of_week
        boolean is_weekend
    }
    
    DIM_CLUSTERS {
        int id PK
        varchar cluster_name
        varchar industry
        text description
    }
```

#### Redis Cache Structure

```
# Key Patterns (with TTL)
metrics:article:{article_id}:realtime (5 min)
metrics:cluster:{cluster_id}:realtime (5 min)
dashboard:demand_trends:{hash} (5 min)
dashboard:competitor_comparison:{hash} (5 min)
alerts:reputation:{article_id} (1 hour)
alerts:hype:{article_id} (1 hour)
ratelimit:wikimedia:{timestamp} (1 sec)
pipeline:status:{pipeline_id} (24 hours)
pipeline:checkpoint:{pipeline_id} (24 hours)
```

### Analytics Layer

The Analytics Layer applies statistical and machine learning models to generate insights.

#### Component Architecture

```mermaid
graph TB
    subgraph "Analytics Components"
        TS[Time Series Forecaster]
        CLUSTER[Topic Clustering Engine]
        HYPE[Hype Detection Engine]
        REP[Reputation Monitor]
        KG[Knowledge Graph Builder]
    end
    
    subgraph "Models & Algorithms"
        PROPHET[Prophet Model]
        ARIMA[ARIMA Model]
        KMEANS[K-Means Clustering]
        TFIDF[TF-IDF Vectorization]
        LOUVAIN[Louvain Algorithm]
        CENTRALITY[Centrality Metrics]
    end
    
    DW[(Data Warehouse)] --> TS
    DW --> CLUSTER
    DW --> HYPE
    DW --> REP
    DW --> KG
    
    TS --> PROPHET
    TS --> ARIMA
    CLUSTER --> TFIDF
    CLUSTER --> KMEANS
    KG --> LOUVAIN
    KG --> CENTRALITY
```

#### Analytics Workflows

**Forecasting Workflow**
```mermaid
sequenceDiagram
    participant DW as Data Warehouse
    participant TS as Time Series Forecaster
    participant CACHE as Redis Cache
    participant DASH as Dashboard
    
    DASH->>CACHE: Check forecast cache
    CACHE-->>DASH: Cache miss
    DASH->>TS: Request forecast
    TS->>DW: Fetch historical data
    DW-->>TS: 180 days of pageviews
    TS->>TS: Train Prophet model
    TS->>TS: Generate 30-day forecast
    TS->>TS: Calculate confidence intervals
    TS->>CACHE: Cache forecast (7 days)
    TS-->>DASH: Return forecast
```

**Reputation Monitoring Workflow**
```mermaid
sequenceDiagram
    participant SCHED as Scheduler
    participant REP as Reputation Monitor
    participant DW as Data Warehouse
    participant ALERT as Alert System
    
    SCHED->>REP: Hourly check
    REP->>DW: Fetch recent edits
    DW-->>REP: Edit history (24h)
    REP->>REP: Calculate edit velocity
    REP->>REP: Detect vandalism signals
    REP->>REP: Calculate risk score
    alt Risk > 0.7
        REP->>ALERT: Generate high-priority alert
        ALERT->>ALERT: Send email notification
        ALERT->>ALERT: Post to webhook
    end
```

### Visualization Layer

The Visualization Layer provides interactive dashboards and reporting capabilities.

#### Dashboard Architecture

```mermaid
graph TB
    subgraph "Dashboard Components"
        SIDEBAR[Sidebar Filters]
        TRENDS[Demand Trends Chart]
        COMPARE[Competitor Table]
        ALERTS[Alerts Panel]
        HEATMAP[Topic Heatmap]
        LEADER[Traffic Leaderboard]
    end
    
    subgraph "Data Sources"
        CACHE[(Redis Cache)]
        DB[(PostgreSQL)]
    end
    
    SIDEBAR --> TRENDS
    SIDEBAR --> COMPARE
    SIDEBAR --> HEATMAP
    SIDEBAR --> LEADER
    
    TRENDS --> CACHE
    COMPARE --> CACHE
    ALERTS --> DB
    HEATMAP --> CACHE
    LEADER --> CACHE
    
    CACHE -.->|Cache miss| DB
```

#### Dashboard Features

- **Auto-refresh**: Configurable interval (default 5 minutes)
- **Filtering**: Date range, industry, metric type
- **Sorting**: All tables support multi-column sorting
- **Export**: CSV and PDF formats
- **Interactivity**: Zoom, pan, tooltips on all charts

### Orchestration & Monitoring

#### Job Scheduling

```mermaid
gantt
    title Scheduled Jobs Timeline
    dateFormat HH:mm
    axisFormat %H:%M
    
    section Hourly
    Pageview Collection :active, h1, 00:00, 1h
    Pageview Collection :h2, 01:00, 1h
    Pageview Collection :h3, 02:00, 1h
    
    section Daily
    Edit History Scraping :d1, 02:00, 2h
    Deep Crawl :d2, 04:00, 3h
    Metrics Aggregation :d3, 07:00, 1h
    
    section Weekly
    Model Retraining :w1, 03:00, 4h
    Data Cleanup :w2, 07:00, 1h
```

#### Health Monitoring

```mermaid
graph LR
    subgraph "Health Checks"
        DB_CHECK[Database Connectivity]
        REDIS_CHECK[Redis Connectivity]
        API_CHECK[API Availability]
        DISK_CHECK[Disk Space]
        MEM_CHECK[Memory Usage]
    end
    
    subgraph "Metrics Collection"
        INGEST_RATE[Ingestion Rate]
        PROCESS_LAT[Processing Latency]
        STORAGE_UTIL[Storage Utilization]
        API_USAGE[API Usage]
        PIPELINE_HEALTH[Pipeline Health]
    end
    
    subgraph "Alerting"
        EMAIL[Email Alerts]
        WEBHOOK[Webhook Notifications]
        LOGS[Structured Logs]
    end
    
    DB_CHECK --> LOGS
    REDIS_CHECK --> LOGS
    API_CHECK --> LOGS
    
    INGEST_RATE --> LOGS
    PROCESS_LAT --> LOGS
    
    LOGS -->|Critical| EMAIL
    LOGS -->|Critical| WEBHOOK
```

## Data Flow

### End-to-End Data Flow

```mermaid
flowchart TD
    START([User Request]) --> DASH[Dashboard]
    DASH --> CACHE{Cache Hit?}
    CACHE -->|Yes| RETURN[Return Cached Data]
    CACHE -->|No| ANALYTICS[Analytics Layer]
    
    ANALYTICS --> DW[(Data Warehouse)]
    DW --> PROCESS{Data Available?}
    
    PROCESS -->|Yes| COMPUTE[Compute Metrics]
    PROCESS -->|No| TRIGGER[Trigger Collection]
    
    TRIGGER --> INGEST[Data Ingestion]
    INGEST --> API[External APIs]
    API --> RAW[Raw Data]
    RAW --> ETL[ETL Pipeline]
    ETL --> VALIDATE{Valid?}
    
    VALIDATE -->|Yes| LOAD[Load to DW]
    VALIDATE -->|No| QUARANTINE[Quarantine]
    
    LOAD --> DW
    COMPUTE --> CACHE_STORE[Store in Cache]
    CACHE_STORE --> RETURN
    
    RETURN --> DASH
    DASH --> END([Display to User])
```

### Real-Time Alert Flow

```mermaid
sequenceDiagram
    participant SCHED as Scheduler
    participant INGEST as Data Ingestion
    participant ETL as ETL Pipeline
    participant DW as Data Warehouse
    participant ANALYTICS as Analytics
    participant ALERT as Alert System
    participant USER as User
    
    SCHED->>INGEST: Trigger hourly collection
    INGEST->>ETL: Send raw data
    ETL->>DW: Load validated data
    SCHED->>ANALYTICS: Trigger analysis
    ANALYTICS->>DW: Query recent data
    DW-->>ANALYTICS: Return metrics
    ANALYTICS->>ANALYTICS: Calculate risk scores
    alt High Risk Detected
        ANALYTICS->>ALERT: Generate alert
        ALERT->>USER: Send email
        ALERT->>USER: Post to webhook
        ALERT->>DW: Log alert
    end
```

## Deployment Architecture

### Docker Deployment

```mermaid
graph TB
    subgraph "Docker Compose Stack"
        subgraph "Application Container"
            APP[Python Application]
            STREAMLIT[Streamlit Server]
            SCHEDULER[Job Scheduler]
        end
        
        subgraph "Database Container"
            POSTGRES[(PostgreSQL 14)]
        end
        
        subgraph "Cache Container"
            REDIS[(Redis 7)]
        end
    end
    
    subgraph "External Services"
        WIKIMEDIA[Wikimedia APIs]
    end
    
    APP --> POSTGRES
    APP --> REDIS
    APP --> WIKIMEDIA
    STREAMLIT --> APP
    SCHEDULER --> APP
```

### Production Deployment

```mermaid
graph TB
    subgraph "Load Balancer"
        LB[Nginx / HAProxy]
    end
    
    subgraph "Application Tier"
        APP1[App Instance 1]
        APP2[App Instance 2]
        APP3[App Instance 3]
    end
    
    subgraph "Data Tier"
        PG_PRIMARY[(PostgreSQL Primary)]
        PG_REPLICA[(PostgreSQL Replica)]
        REDIS_CLUSTER[(Redis Cluster)]
    end
    
    subgraph "Monitoring"
        PROMETHEUS[Prometheus]
        GRAFANA[Grafana]
    end
    
    LB --> APP1
    LB --> APP2
    LB --> APP3
    
    APP1 --> PG_PRIMARY
    APP2 --> PG_PRIMARY
    APP3 --> PG_PRIMARY
    
    APP1 --> REDIS_CLUSTER
    APP2 --> REDIS_CLUSTER
    APP3 --> REDIS_CLUSTER
    
    PG_PRIMARY --> PG_REPLICA
    
    APP1 --> PROMETHEUS
    APP2 --> PROMETHEUS
    APP3 --> PROMETHEUS
    PROMETHEUS --> GRAFANA
```

## Scalability Considerations

### Horizontal Scaling

- **Application Tier**: Stateless design allows adding workers
- **Data Ingestion**: Distributed crawling across multiple processes
- **Analytics**: Parallel processing of independent articles/clusters
- **Database**: Read replicas for query load distribution

### Performance Optimization

- **Caching Strategy**: Redis for hot data (5-minute TTL)
- **Query Optimization**: Indexed columns, partitioned tables
- **Async I/O**: All network operations use async/await
- **Connection Pooling**: Reuse database connections
- **Lazy Loading**: Dashboard loads data on-demand

### Capacity Planning

| Component | Current | Target | Scaling Strategy |
|-----------|---------|--------|------------------|
| Pageviews/day | 1M | 10M | Add ingestion workers |
| Articles tracked | 10K | 100K | Partition by article_id |
| Dashboard users | 10 | 100 | Add app instances |
| Forecast models | 100 | 1000 | Distributed training |
| Storage | 100GB | 1TB | Table partitioning |

## Security Architecture

### Authentication & Authorization

```mermaid
graph LR
    USER[User] --> AUTH[Authentication]
    AUTH --> JWT[JWT Token]
    JWT --> API[API Gateway]
    API --> RBAC[Role-Based Access]
    RBAC --> RESOURCES[Protected Resources]
```

### Data Security

- **Encryption at Rest**: Database encryption enabled
- **Encryption in Transit**: TLS for all API calls
- **Secrets Management**: Environment variables, encrypted config
- **API Keys**: Stored encrypted, rotated regularly
- **Audit Logging**: All access logged with timestamps

## Disaster Recovery

### Backup Strategy

- **Database**: Daily full backups, hourly incrementals
- **Redis**: Persistence enabled (RDB + AOF)
- **Configuration**: Version controlled in Git
- **Logs**: Retained for 30 days, archived to S3

### Recovery Procedures

1. **Database Failure**: Promote replica to primary
2. **Cache Failure**: Fallback to database queries
3. **Application Failure**: Auto-restart with health checks
4. **Data Corruption**: Restore from latest backup

## Monitoring & Observability

### Key Metrics

- **Ingestion**: Records/second, API latency, error rate
- **Processing**: Pipeline duration, validation failures, quarantine rate
- **Storage**: Query latency, connection pool usage, disk utilization
- **Analytics**: Model training time, prediction accuracy, cache hit rate
- **Dashboard**: Page load time, concurrent users, export requests

### Logging Strategy

- **Format**: Structured JSON logs
- **Levels**: ERROR, WARNING, INFO, DEBUG
- **Retention**: 30 days in system, archived to S3
- **Aggregation**: Centralized logging with ELK stack

## Technology Stack

| Layer | Technologies |
|-------|-------------|
| Language | Python 3.11+ |
| Web Framework | Streamlit |
| Database | PostgreSQL 14+ |
| Cache | Redis 7+ |
| ML/Analytics | Prophet, scikit-learn, NetworkX |
| Data Processing | pandas, numpy |
| Web Crawling | Crawl4AI, aiohttp |
| Testing | pytest, Hypothesis |
| Orchestration | APScheduler |
| Containerization | Docker, Docker Compose |
| Migrations | Alembic |

## Future Enhancements

1. **Machine Learning**
   - Deep learning for trend prediction
   - NLP for sentiment analysis
   - Anomaly detection with autoencoders

2. **Scalability**
   - Kubernetes deployment
   - Distributed task queue (Celery)
   - Sharded database architecture

3. **Features**
   - Real-time streaming analytics
   - Custom alert rules engine
   - Multi-language support
   - Mobile app

4. **Integration**
   - Slack/Teams notifications
   - Jira ticket creation
   - Salesforce integration
   - Custom webhooks
