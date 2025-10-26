-- Cell information
CREATE TABLE cells (
    cell_id TEXT PRIMARY KEY,
    province_name TEXT,
    district_name TEXT,
    sector_name TEXT,
    cell_name TEXT
);

-- Population 
CREATE TABLE pop (
    cell_id TEXT PRIMARY KEY,
    elderly_60 REAL,
    general_pop REAL,
    children_under5 REAL,
    youth_15_24 REAL,
    men_2020 REAL,
    women_2020 REAL,
    building_count REAL
);

-- Annual NTL stats
CREATE TABLE ntl_annual (
    id SERIAL PRIMARY KEY,
    cell_id TEXT REFERENCES cells(cell_id),
    ntl_min REAL,
    ntl_max REAL,
    ntl_mean REAL,
    ntl_median REAL,
    ntl_sum REAL,
    pixel_count REAL,
    raster_filename TEXT
);


-- Monthly NTL stats
CREATE TABLE ntl_monthly (
    id SERIAL PRIMARY KEY,
    cell_id TEXT REFERENCES cells(cell_id),
    ntl_min REAL,
    ntl_max REAL,
    ntl_mean REAL,
    ntl_median REAL,
    ntl_sum REAL,
    pixel_count REAL,
    raster_filename TEXT,
    year INT,
    month INT,
    date DATE
);


-- LOAD DATA FROM CSV FILES
--COPY cells FROM '/Users/dmatekenya/My Drive (dmatekenya@gmail.com)/TEACHING/AIMS-DSCBI/data/tmp-db-data/cells.csv' DELIMITER ',' CSV HEADER;
\copy cells FROM '/home/risa/Desktop/RISA/TRAININGS/DSCBI/Python Class/projects/AIMS-DSCBI/tmp-db-data-20250915T092258Z-1-001/tmp-db-data/cells.csv' DELIMITER ',' CSV HEADER;
\copy pop (cell_id, elderly_60, children_under5, youth_15_24, general_pop, men_2020, women_2020, building_count)FROM '/home/risa/Desktop/RISA/TRAININGS/DSCBI/Python Class/projects/AIMS-DSCBI/tmp-db-data-20250915T092258Z-1-001/tmp-db-data/population.csv' DELIMITER ',' CSV HEADER;
\copy ntl_annual (ntl_annual (cell_id, ntl_min, ntl_max, ntl_mean, ntl_median, ntl_sum,pixel_count, raster_filename, year)FROM '/home/risa/Desktop/RISA/TRAININGS/DSCBI/Python Class/projects/AIMS-DSCBI/tmp-db-data-20250915T092258Z-1-001/tmp-db-data/ntl-annual-2012-2024.csv' DELIMITER ',' CSV HEADER;
\copy ntl_monthly(cell_id, ntl_min, ntl_max, ntl_mean, ntl_median, ntl_sum, pixel_count, raster_filename, year, month, date)
FROM '/home/risa/Desktop/RISA/TRAININGS/DSCBI/Python Class/projects/AIMS-DSCBI/tmp-db-data-20250915T092258Z-1-001/tmp-db-data/ntl-monthly-2012-2024.csv'
DELIMITER ',' CSV HEADER;
\copy ntl_annual FROM '/home/risa/Desktop/RISA/TRAININGS/DSCBI/Python Class/projects/AIMS-DSCBI/tmp-db-data-20250915T092258Z-1-001/tmp-db-data/ntl-annual-2012-2024.csv' DELIMITER ',' CSV HEADER;




--BASIC QUERIES 
-- Check average light in a given year
SELECT year, AVG(ntl_mean)
FROM ntl_annual
GROUP BY year
ORDER BY year;

-- Get top 5 brightest cells in 2023
SELECT cell_id, ntl_sum
FROM ntl_annual
WHERE year = 2023
ORDER BY ntl_sum DESC
LIMIT 5;


-- SELECT c.province_name,c.district,c.sector,c.cell_name,n.ntl_sum FROM ntl_annual a  JOIN cells c ON a.cell_id = c.cell_id WHERE n.year = 2023 AND n.ntl_sum > 50 ORDER BY n.ntl_sum DESC;