// Create uniqueness constraints for node identifiers
// Constraints automatically create indexes

// Company node constraints
CREATE CONSTRAINT company_id_unique IF NOT EXISTS
FOR (c:Company) REQUIRE c.id IS UNIQUE;

// Repository node constraints
CREATE CONSTRAINT repository_id_unique IF NOT EXISTS
FOR (r:Repository) REQUIRE r.id IS UNIQUE;

// Sector node constraints
CREATE CONSTRAINT sector_id_unique IF NOT EXISTS
FOR (s:Sector) REQUIRE s.id IS UNIQUE;
