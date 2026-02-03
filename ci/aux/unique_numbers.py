import sys

def main():

    if len(sys.argv) != 2:
        print("Usage: unique_numbers.py <comma-separated-numbers-without-whitespaces>", file=sys.stderr)
        sys.exit(1)

    raw_input = sys.argv[1]

    numbers = raw_input.split(",")

    unique_numbers = set()

    for number in numbers:
        if not number.isdigit():
            print(
                f"Error: invalid natural number '{number}'",
                file=sys.stderr
            )
            sys.exit(1)

        number = int(number)
        unique_numbers.add(number)

    output = ",".join(str(num) for num in sorted(unique_numbers))
    print(output)
    return

main()