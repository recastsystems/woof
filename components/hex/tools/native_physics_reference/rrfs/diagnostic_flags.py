"""Named compiler-only diagnostics; never flags for a primary reference."""

INTEGER_POISON = -2147483647
DIAGNOSTICS = {
    'bounds-snan': ('-g', '-fcheck=all', '-finit-real=snan'),
    'bounds-snan-integer-poison': (
        '-g', '-fcheck=all', '-finit-real=snan', f'-finit-integer={INTEGER_POISON}'),
}


def checked_flags(diagnostic):
    try:
        return DIAGNOSTICS[diagnostic]
    except KeyError:
        raise ValueError('unsupported native diagnostic: ' + str(diagnostic)) from None
