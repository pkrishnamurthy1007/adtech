def extract_octets(ip, octets=None, join=False):
    # extract_octets('17.198.0.0', [1,2], join=True)
    # >>> '17.198'
    # extract_octets('17.198.0.0', [1,2])
    # >>> ['17'.'198']
    # extract_octets('17.198.0.0')
    # >>> ['17', '198', '0', '0']

    if octets is None: octets = [1, 2, 3, 4]
    split_ip = ip.split('.')
    ip_parts = [split_ip[octet - 1] for octet in octets]

    if join: ip_parts = '.'.join(ip_parts)

    return ip_parts
