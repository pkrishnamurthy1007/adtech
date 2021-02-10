import csv
from ipaddress import IPv4Network, IPv4Address
from generic import Timer
from geoip.ip_utils import extract_octets


# Takes the maxmind blocks file and adds 3 columns that aid in optimizing database joins on redshift:
#   ip_index - first two octets of the ipv4 address
#   start_int - integer representation of the first ip in the range
#   end_int - integer representation of the last ip in the range

# debugging option, stop at this many rows for testing, None to disable
breakrows = None
infile = '/datascience-utils/geoip/maxmind20200901/GeoIP2-City-Blocks-IPv4.csv'
outfile = '/datascience-utils/geoip/maxmind20200901/GeoIP2-City-Blocks-IPv4-opt.csv'

if __name__ == '__main__':
    timer = Timer()
    timer.set_timer()

    with open(infile, 'r') as r, open(outfile, 'w') as o:
        reader = csv.reader(r)
        writer = csv.writer(o)
        writer.writerow(['ip_index', 'start_int', 'end_int'] + next(reader))

        breakrow = 0
        for row in reader:

            network = IPv4Network(row[0])
            start_ip, end_ip = str(network[0]), str(network[-1])

            network_first_octet = extract_octets(start_ip, octets=[1], join=True)
            network_end_octet = extract_octets(start_ip, octets=[1], join=True)
            if network_first_octet != network_end_octet:
                raise ValueError("the first octet of the start and end ip do not match, this scenario is unsupported")

            first_second_octet = extract_octets(start_ip, octets=[2], join=True)
            last_second_octet = extract_octets(end_ip, octets=[2], join=True)

            ip_indexes = list()

            # if the first and last ip in the range have the same first two octets, no additional rows are needed
            if int(first_second_octet) == int(last_second_octet):
                ix = '.'.join([network_first_octet, first_second_octet])
                ip_indexes.append([ix, int(network[0]), int(network[-1])])

            # if the range of ips covers several second octets, i.e. 30.41 to 30.45, calculate all the start
            # and end integer ranges
            else:
                octet = int(first_second_octet)
                octets = [str(octet)]
                while octet < int(last_second_octet):
                    octet += 1
                    octets.append(str(octet))

                first_range_start_ip_int = int(network[0])

                first_range_end_ip = '.'.join([network_first_octet, first_second_octet, '255.255'])
                first_range_end_ip_int = int(IPv4Address(first_range_end_ip))

                first_ip_ix = '.'.join([network_first_octet, first_second_octet])

                ip_indexes.append([first_ip_ix, first_range_start_ip_int, first_range_end_ip_int])

                for second_octet in octets[1:-1]:
                    ip_ix = '.'.join([network_first_octet, second_octet])

                    first_ip = '.'.join([network_first_octet, second_octet, '0.0'])
                    last_ip = '.'.join([network_first_octet, second_octet, '255.255'])
                    first_ip_int, last_ip_int = int(IPv4Address(first_ip)), int(IPv4Address(last_ip))

                    ip_indexes.append([ip_ix, first_ip_int, last_ip_int])

                end_range_start_ip = '.'.join([network_first_octet, last_second_octet, '0.0'])
                end_range_start_ip_int = int(IPv4Address(end_range_start_ip))

                end_range_end_ip_int = int(network[-1])

                last_ip_ix = '.'.join([network_first_octet, last_second_octet])

                ip_indexes.append([last_ip_ix, end_range_start_ip_int, end_range_end_ip_int])

            for ix in ip_indexes:
                writer.writerow(ix + row)

            breakrow += 1

            if breakrow in [(n + 1) * 1000000 for n in range(10)]:
                print(breakrow, timer.seconds_elapsed(textout=True))
            if breakrows is not None and breakrow > breakrows: break

    speed = timer.seconds_elapsed(textout=False)

    print(speed)
    print(str(breakrow / speed) + ' rows per second')


