import argparse
import sys

def main():
    # Create the parser
    parser = argparse.ArgumentParser(
        description="A simple example using argparse in Python."
    )

    # Add positional argument
    parser.add_argument(
        "name",
        type=str,
        help="Your name (string)"
    )

    # Add optional argument with default value
    parser.add_argument(
        "-a", "--age",
        type=int,
        default=None,
        help="Your age (integer)"
    )

    # Add a flag (boolean switch)
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    # Parse the arguments
    args = parser.parse_args()

    # Validate age if provided
    if args.age is not None and args.age < 0:
        print("Error: Age cannot be negative.", file=sys.stderr)
        sys.exit(1)

    # Output
    greeting = f"Hello, {args.name}!"
    if args.age is not None:
        greeting += f" You are {args.age} years old."
    print(greeting)

    if args.verbose:
        print("Verbose mode is ON.")

if __name__ == "__main__":
    main()
