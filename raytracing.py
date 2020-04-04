import numpy as np
import matplotlib.pyplot as plt

W = 400
H = 300

# Calculating the intersection of the ray, starting from some point of the object's surface, with other objects,
# we need to offset the ray's start point by normal from the surface.
# Otherwise the ray may intersect current object in it's starting point.
SURFACE_OFFSET = .0001


def normalize(x):
    x /= np.linalg.norm(x)
    return x


def intersect_plane(O, D, P, N):
    # Return the distance from O to the intersection of the ray (O, D) with the
    # plane (P, N), or +inf if there is no intersection.
    # O and P are 3D points, D and N (normal) are normalized vectors.
    denom = np.dot(D, N)
    if np.abs(denom) < 1e-6:
        return np.inf
    d = np.dot(P - O, N) / denom
    if d < 0:
        return np.inf
    return d


def intersect_sphere(O, D, S, R):
    # Return the distance from O to the intersection of the ray (O, D) with the
    # sphere (S, R), or +inf if there is no intersection.
    # O and S are 3D points, D (direction) is a normalized vector, R is a scalar.
    a = np.dot(D, D)
    OS = O - S
    b = 2 * np.dot(D, OS)
    c = np.dot(OS, OS) - R * R
    disc = b * b - 4 * a * c
    if disc > 0:
        dist_sqrt = np.sqrt(disc)
        q = (-b - dist_sqrt) / 2.0 if b < 0 else (-b + dist_sqrt) / 2.0
        t0 = q / a
        t1 = c / q
        t0, t1 = min(t0, t1), max(t0, t1)
        if t1 >= 0:
            return t1 if t0 < 0 else t0
    return np.inf


def intersect(O, D, obj):
    if obj['type'] == 'plane':
        return intersect_plane(O, D, obj['position'], obj['normal'])
    elif obj['type'] == 'sphere':
        return intersect_sphere(O, D, obj['position'], obj['radius'])


def get_normal(obj, M):
    # Find normal.
    if obj['type'] == 'sphere':
        N = normalize(M - obj['position'])
    elif obj['type'] == 'plane':
        N = obj['normal']
    return N


def get_color(obj, M):
    color = obj['color']
    if not hasattr(color, '__len__'):
        color = color(M)
    return color


def trace_ray(rayO, rayD):
    # Find first point of intersection with the scene.
    t = np.inf
    for i, obj in enumerate(scene):
        t_obj = intersect(rayO, rayD, obj)
        if t_obj < t:
            t, obj_idx = t_obj, i
    # Return None if the ray does not intersect any object.
    if t == np.inf:
        return
    # Find the object.
    obj = scene[obj_idx]
    # Find the point of intersection on the object.
    M = rayO + rayD * t
    # Find properties of the object.
    N = get_normal(obj, M)
    color = get_color(obj, M)
    toL = normalize(L - M)
    toO = normalize(O - M)
    # Shadow: find if the point is shadowed or not.
    l = [intersect(M + N * SURFACE_OFFSET, toL, obj_sh) for k, obj_sh in enumerate(scene) if k != obj_idx]
    if l and min(l) < np.inf:
        return
    # Start computing the color.
    col_ray = ambient
    # Lambert shading (diffuse).
    col_ray += obj.get('diffuse_c', diffuse_c) * max(np.dot(N, toL), 0) * color
    # Blinn-Phong shading (specular).
    col_ray += obj.get('specular_c', specular_c) * max(np.dot(N, normalize(toL + toO)), 0) ** specular_k * color_light
    return obj, M, N, col_ray


def reflection(rayD, normal):
    return normalize(rayD - 2. * np.dot(rayD, normal) * normal)


def refraction(rayD, normal, refr_cff):
    c1 = - np.dot(rayD, normal)
    c2_squared = 1. - refr_cff ** 2 * (1. - c1 ** 2)
    # total internal reflection
    if c2_squared < 0.:
        return
    c2 = np.math.sqrt(c2_squared)
    return refr_cff * rayD + (refr_cff * c1 - c2) * normal


def trace_ray_rec(rayO, rayD, light_intensity, depth):
    if depth > depth_max or light_intensity == 0.:
        return np.zeros(3)

    tracing_result = trace_ray(rayO, rayD)
    if not tracing_result:
        return np.zeros(3)
    obj, M, normal, col_ray = tracing_result
    color_result = col_ray * light_intensity

    refr_cff = 1. / obj['refraction']
    inside_surface = np.dot(rayD, normal)
    if inside_surface > 0.:
        # ray goes from the inside of the surface
        normal = -normal
        refr_cff = 1. / refr_cff

    # create reflected ray
    refl_rayO = M + normal * SURFACE_OFFSET
    refl_rayD = reflection(rayD, normal)
    color_result += trace_ray_rec(refl_rayO, refl_rayD, light_intensity * obj['reflection'], depth + 1)

    # create refracted ray
    refr_rayO = M - normal * SURFACE_OFFSET
    refr_rayD = refraction(rayD, normal, refr_cff)
    if refr_rayD is not None:
        color_result += trace_ray_rec(refr_rayO, refr_rayD, light_intensity * obj['transparency'], depth + 1)

    return color_result


def add_sphere(position, radius, color, refl_cff=.25, refr_cff=1., transparency=0.):
    return dict(type='sphere', position=np.array(position), radius=np.array(radius), color=np.array(color),
                reflection=refl_cff, refraction=refr_cff, transparency=transparency)


def add_plane(position, normal, refl_cff=.25, refr_cff=1., transparency=0.):
    return dict(type='plane', position=np.array(position), normal=np.array(normal),
                color=lambda M: (color_plane0 if (int(M[0] * 2) % 2) == (int(M[2] * 2) % 2) else color_plane1),
                diffuse_c=.75, specular_c=.5, reflection=refl_cff, refraction=refr_cff, transparency=transparency)


# List of objects.
color_plane0 = 1. * np.ones(3)
color_plane1 = 0. * np.ones(3)
scene = [add_sphere([.75, .1, 1.], .6, [0., 0., 1.], refl_cff=.25, refr_cff=1.3, transparency=.9),
         add_sphere([.75, .1, 1.], .3, [0., 1., 0.], refl_cff=0., refr_cff=1.5, transparency=0.),
         add_sphere([-.75, .1, 2.25], .6, [.5, .223, .5], refl_cff=.25, refr_cff=1., transparency=1.),
         add_sphere([-2.75, .1, 3.5], .6, [1., .572, .184], refl_cff=.25, refr_cff=1.7, transparency=.6),
         add_plane([0., -.5, 0.], [0., 1., 0.], refl_cff=.3, refr_cff=1., transparency=0.),
         ]

# Light position and color.
L = np.array([5., 5., -10.])
color_light = np.ones(3)

# Default light and material parameters.
ambient = .05
diffuse_c = 1.
specular_c = 1.
specular_k = 50

depth_max = 5  # Maximum number of light reflections.
col = np.zeros(3)  # Current color.
O = np.array([0., 0.35, -1.])  # Camera.
Q = np.array([0., 0., 0.])  # Camera pointing to.
img = np.zeros((H, W, 3))

r = float(W) / H
# Screen coordinates: x0, y0, x1, y1.
S = (-1., -1. / r + .25, 1., 1. / r + .25)

# Loop through all pixels.
for i, x in enumerate(np.linspace(S[0], S[2], W)):
    if i % 10 == 0:
        print(i / float(W) * 100, "%")
    for j, y in enumerate(np.linspace(S[1], S[3], H)):
        col[:] = 0
        Q[:2] = (x, y)
        D = normalize(Q - O)
        col = trace_ray_rec(O, D, 1., 1)
        img[H - j - 1, i, :] = np.clip(col, 0, 1)

plt.imsave('fig.png', img)
