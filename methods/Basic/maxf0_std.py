def maxf0_std(AGE_GENDER):
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
    STDNI = max(STDNC, STDXI / 8)
    STDNE = min(STDNM, STDXE / 4)

    if AGE_GENDER == '?':
        print('[maxf0,<minf0>] = maxf0_std(<"n"|AGE_GENDER|"m"|"f"|"i"|"c"|"e">)')
        return

    if AGE_GENDER is None:
        AGE_GENDER = ''

    if len(AGE_GENDER) <= 1:
        age_gender = AGE_GENDER.upper()

        if age_gender == 'M':
            maxf0 = STDXM
        elif age_gender in ('F', 'N'):
            maxf0 = STDXF
        elif age_gender == 'C':
            maxf0 = STDXC
        elif age_gender == 'I':
            maxf0 = STDXI
        elif age_gender == 'E':
            maxf0 = STDXE
        else:
            maxf0 = float('nan')

        if age_gender in ('M', 'N'):
            minf0 = STDNM
        elif age_gender == 'F':
            minf0 = STDNF
        elif age_gender == 'C':
            minf0 = STDNC
        elif age_gender == 'I':
            minf0 = STDNI
        elif age_gender == 'E':
            minf0 = STDNE
        else:
            minf0 = float('nan')

    else:
        maxf01, minf01 = maxf0_std(AGE_GENDER[0])
        maxf0, minf0 = maxf0_std(AGE_GENDER[1:])
        maxf0 = [maxf01] + maxf0
        minf0 = [minf01] + minf0

    return maxf0, minf0
