import warp as wp


@wp.func
def cexp(amplitude: wp.float32, phase: wp.float32) -> wp.vec2:
    """Complex exponential: exp(a + bi) = exp(a) * (cos(b) + i*sin(b))"""

    return wp.vec2(amplitude * wp.cos(phase), amplitude * wp.sin(phase))


@wp.func
def cmul(a: wp.vec2, b: wp.vec2) -> wp.vec2:
    """Complex multiplication: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i"""
    return wp.vec2(a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0])


@wp.func
def cconj(z: wp.vec2) -> wp.vec2:
    """Complex conjugate: (a + bi)* = (a - bi)"""
    return wp.vec2(z[0], -z[1])


@wp.func
def cabs(z: wp.vec2) -> wp.float32:
    """Complex absolute value: |a + bi| = sqrt(a^2 + b^2)"""
    return wp.sqrt(z[0] * z[0] + z[1] * z[1])
