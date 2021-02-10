SELECT
    s.creation_date,
    convert_timezone('UTC', l.time_zone, s.creation_date) AS user_location_datetime,
    s.session_id,
    s.ip_address,
    b.network AS cidr,
    b.is_anonymous_proxy,
    b.is_satellite_provider,
    l.city_name,
    b.postal_code,
    b.latitude,
    b.longitude,
    b.accuracy_radius,
    l.country_iso_code AS country_code,
    l.subdivision_1_iso_code AS state,
    l.time_zone,
    l.metro_code AS dma
FROM tracking.session_detail AS s
LEFT JOIN data_science.maxmind_ipv4_geo_blocks AS b
    ON ip_index(s.ip_address) = b.netowrk_index
    AND inet_aton(s.ip_address) BETWEEN b.start_int AND b.end_int
LEFT JOIN data_science.maxmind_geo_locations AS l
    ON b.maxmind_id = l.maxmind_id
WHERE nullif(s.ip_address, '') IS NOT null
LIMIT 100
;
