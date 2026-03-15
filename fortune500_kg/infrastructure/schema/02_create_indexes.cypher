// Create indexes for query performance optimization

// Company indexes for filtering and sorting
CREATE INDEX company_sector_idx IF NOT EXISTS
FOR (c:Company) ON (c.sector);

CREATE INDEX company_revenue_rank_idx IF NOT EXISTS
FOR (c:Company) ON (c.revenue_rank);

CREATE INDEX company_name_idx IF NOT EXISTS
FOR (c:Company) ON (c.name);

// Repository indexes
CREATE INDEX repository_name_idx IF NOT EXISTS
FOR (r:Repository) ON (r.name);

CREATE INDEX repository_stars_idx IF NOT EXISTS
FOR (r:Repository) ON (r.stars);

// Sector indexes
CREATE INDEX sector_name_idx IF NOT EXISTS
FOR (s:Sector) ON (s.name);

// Composite indexes for common query patterns
CREATE INDEX company_sector_revenue_idx IF NOT EXISTS
FOR (c:Company) ON (c.sector, c.revenue_rank);
