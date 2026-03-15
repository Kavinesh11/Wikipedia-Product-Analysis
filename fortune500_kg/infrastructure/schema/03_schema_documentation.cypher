// Fortune 500 Knowledge Graph Schema Documentation
// This file documents the graph schema structure

// ============================================================================
// NODE TYPES
// ============================================================================

// Company Node
// Represents a Fortune 500 company with business and GitHub metrics
// Properties:
//   - id: String (UNIQUE) - Unique identifier for the company
//   - name: String - Company name
//   - sector: String - Industry sector (indexed)
//   - revenue_rank: Integer - Fortune 500 rank (indexed)
//   - employee_count: Integer - Number of employees
//   - github_org: String - GitHub organization name
//   - created_at: DateTime - Node creation timestamp
//   - updated_at: DateTime - Last update timestamp

// Repository Node
// Represents a GitHub repository owned or used by companies
// Properties:
//   - id: String (UNIQUE) - GitHub repository ID
//   - name: String - Repository name
//   - stars: Integer - GitHub stars count
//   - forks: Integer - GitHub forks count
//   - contributors: Integer - Number of contributors
//   - created_at: DateTime - Repository creation timestamp

// Sector Node
// Represents an industry sector with aggregated metrics
// Properties:
//   - id: String (UNIQUE) - Sector identifier
//   - name: String - Sector name
//   - avg_innovation_score: Float - Average innovation score for sector
//   - avg_digital_maturity: Float - Average digital maturity for sector

// ============================================================================
// RELATIONSHIP TYPES
// ============================================================================

// (:Company)-[:OWNS]->(:Repository)
// Indicates a company owns a GitHub repository
// Properties: None

// (:Company)-[:PARTNERS_WITH]->(:Company)
// Indicates a partnership between two companies
// Properties:
//   - since: Date - Partnership start date
//   - partnership_type: String - Type of partnership

// (:Company)-[:ACQUIRED]->(:Company)
// Indicates one company acquired another
// Properties:
//   - date: Date - Acquisition date
//   - amount: Float - Acquisition amount in USD

// (:Company)-[:BELONGS_TO]->(:Sector)
// Indicates a company belongs to an industry sector
// Properties: None

// (:Company)-[:DEPENDS_ON]->(:Repository)
// Indicates a company has a technology dependency on a repository
// Properties:
//   - dependency_type: String - Type of dependency (e.g., "direct", "transitive")

// ============================================================================
// EXAMPLE QUERIES
// ============================================================================

// Find all companies in a specific sector
// MATCH (c:Company)-[:BELONGS_TO]->(s:Sector {name: 'Technology'})
// RETURN c.name, c.revenue_rank
// ORDER BY c.revenue_rank;

// Find companies with high GitHub activity
// MATCH (c:Company)-[:OWNS]->(r:Repository)
// WITH c, sum(r.stars) as total_stars, sum(r.forks) as total_forks
// WHERE total_stars > 10000
// RETURN c.name, total_stars, total_forks
// ORDER BY total_stars DESC;

// Find partnership networks
// MATCH (c1:Company)-[p:PARTNERS_WITH]->(c2:Company)
// RETURN c1.name, c2.name, p.partnership_type, p.since
// ORDER BY p.since DESC;

// Find acquisition history
// MATCH (acquirer:Company)-[a:ACQUIRED]->(target:Company)
// RETURN acquirer.name, target.name, a.date, a.amount
// ORDER BY a.date DESC;
