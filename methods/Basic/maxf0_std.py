def maxf0_std(AGE_GENDER):
    # Flatten the input list
    AGE_GENDER = [char for string in AGE_GENDER for char in string]

    # Function to process a single character
    def maxf0_std_single_character(x):
        if x == "?":
            print("[maxf0, minf0] = maxf0_std(<'n'|AGE_GENDER|'m'|'f'|'i'|'c'|'e'>)")
            return

        if not x:
            x = "N"  # Default = "N": adult, no gender-specific information

        # Max's:
        STDXM = 220
        STDXF = 350
        STDXC = 500
        STDXI = 1200
        STDXE = (STDXM + STDXF) / 2

        # Min's:
        STDNM = STDXM / 4
        STDNF = STDXF / 4
        STDNC = STDXC / 5

        # 08/9/30: Was: STDNI = STDXI / 8; STDNE = STDXE / 4;
        STDNI = max(STDNC, STDXI / 8)
        STDNE = min(STDNM, STDXE / 4)

        # Convert input character to upper case (case-insensitive)
        x = x.upper()

        if x == "M":
            maxf0 = STDXM
        elif x in {"F", "N"}:
            maxf0 = STDXF  # "N" => no info => use higher for max.
        elif x == "C":
            maxf0 = STDXC
        elif x == "I":
            maxf0 = STDXI
        elif x == "E":
            maxf0 = STDXE
        else:
            maxf0 = None

        if x in {"M", "N"}:
            minf0 = STDNM  # "N" => no info => use lowest for min.
        elif x == "F":
            minf0 = STDNF
        elif x == "C":
            minf0 = STDNC
        elif x == "I":
            minf0 = STDNI
        elif x == "E":
            minf0 = STDNE
        else:
            minf0 = None

        return maxf0, minf0

    # Apply the function to each character in the list
    record = list(map(maxf0_std_single_character,AGE_GENDER))

    # Separate the maxf0 and minf0 values into separate lists
    maxf0 = [item[0] for item in record]
    minf0 = [item[1] for item in record]

    return maxf0, minf0

# def test_maxf0_std():

#     # Example 1
#     x1 = maxf0_std('mf')
#     print(x1)

#     # Example 2
#     x2 = maxf0_std(['m', 'f'])
#     print(x2)

#     # Example 3
#     x3 = maxf0_std(['mfi', 'c', 'e'])
#     print(x3)

# # Run the test code
# test_maxf0_std()