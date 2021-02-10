--/
CREATE FUNCTION inet_aton(VARCHAR)
RETURNS BIGINT IMMUTABLE AS $$
SELECT (
    (split_part($1, '.', 1)::BIGINT << 24) +
    (split_part($1, '.', 2)::BIGINT << 16) +
    (split_part($1, '.', 3)::BIGINT << 8) +
    (split_part($1, '.', 4)::BIGINT)
    )
$$ LANGUAGE sql;
/

--/
CREATE FUNCTION ip_index(VARCHAR)
RETURNS VARCHAR IMMUTABLE AS $$
SELECT regexp_substr($1, '\\d+\.\\d+')
$$ LANGUAGE sql;
/

