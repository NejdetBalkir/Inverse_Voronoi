import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from math import atan2

# ------------------- tunables -------------------
THRESH_BIN = 200          # binarization threshold
NOISE_MIN_BRANCH = 3      # prune spur branches shorter than this (px)
MERGE_VERT_DIST = 3       # cluster tolerance for coincident vertices (px)
# ------------------------------------------------

# ---------- basic helpers ----------
def to_binary_inv(gray):
    _, th = cv2.threshold(gray, THRESH_BIN, 255, cv2.THRESH_BINARY_INV)
    return (th // 255).astype(np.uint8)  # 0/1

def skeletonize_image(gray):
    return skeletonize(to_binary_inv(gray)).astype(np.uint8)  # 0/1

NB8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
def neighbors8(y,x, sk):
    h,w = sk.shape
    out=[]
    for dy,dx in NB8:
        yy,xx = y+dy, x+dx
        if 0<=yy<h and 0<=xx<w and sk[yy,xx]==1:
            out.append((yy,xx))
    return out

def prune_spurs(skel, keep_border=True):
    sk = skel.copy()
    h,w = sk.shape
    changed = True
    while changed:
        changed=False
        to_del=[]
        for y in range(h):
            for x in range(w):
                if sk[y,x]==1 and len(neighbors8(y,x,sk))==1:
                    if keep_border and (x==0 or y==0 or x==w-1 or y==h-1):
                        continue
                    # walk along degree-2 path; if short -> delete
                    path=[(y,x)]
                    py,px = y,x
                    ny,nx = neighbors8(y,x,sk)[0]
                    while sk[ny,nx]==1 and len(neighbors8(ny,nx,sk))==2 and len(path)<NOISE_MIN_BRANCH+1:
                        nbs = neighbors8(ny,nx,sk)
                        nxt = nbs[0] if nbs[0]!=(py,px) else (nbs[1] if len(nbs)>1 else nbs[0])
                        path.append((ny,nx))
                        py,px = ny,nx
                        ny,nx = nxt
                    if len(path)<=NOISE_MIN_BRANCH:
                        to_del.extend(path)
        if to_del:
            changed=True
            for (yy,xx) in to_del:
                sk[yy,xx]=0
    return sk

def detect_rectangle_via_hough(gray):
    """Return (xL,yT,xR,yB) and corner points; fall back to image bounds."""
    h,w = gray.shape
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80,
                            minLineLength=int(0.6*min(h,w)), maxLineGap=10)
    if lines is None:
        return (0.0,0.0,w-1.0,h-1.0), [(0.0,0.0),(w-1.0,0.0),(w-1.0,h-1.0),(0.0,h-1.0)]
    horizontals=[]; verticals=[]
    for l in lines[:,0]:
        x1,y1,x2,y2 = l
        if abs(y2-y1) <= abs(x2-x1): horizontals.append(l)
        else: verticals.append(l)
    if not horizontals or not verticals:
        return (0.0,0.0,w-1.0,h-1.0), [(0.0,0.0),(w-1.0,0.0),(w-1.0,h-1.0),(0.0,h-1.0)]
    top = min(horizontals, key=lambda L: min(L[1],L[3]))
    bot = max(horizontals, key=lambda L: max(L[1],L[3]))
    lef = min(verticals,   key=lambda L: min(L[0],L[2]))
    rig = max(verticals,   key=lambda L: max(L[0],L[2]))
    yT = 0.5*(top[1]+top[3]); yB = 0.5*(bot[1]+bot[3])
    xL = 0.5*(lef[0]+lef[2]); xR = 0.5*(rig[0]+rig[2])
    corners = [(xL,yT),(xR,yT),(xR,yB),(xL,yB)]
    return (xL,yT,xR,yB), corners

def line_intersection(p1,p2, q1,q2):
    x1,y1 = p1; x2,y2=p2; x3,y3=q1; x4,y4=q2
    den = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(den) < 1e-9: return None
    px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4))/den
    py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4))/den
    return (px,py)

def project_to_rectangle(p_prev, p_end, rect):
    xL,yT,xR,yB = rect
    walls = [((xL,yT),(xR,yT)), ((xR,yT),(xR,yB)),
             ((xR,yB),(xL,yB)), ((xL,yB),(xL,yT))]
    for a,b in walls:
        ip = line_intersection(p_prev, p_end, a, b)
        if ip is None: continue
        x,y=ip
        if xL-1e-6 <= x <= xR+1e-6 and yT-1e-6 <= y <= yB+1e-6:
            return (x,y)
    return None

def merge_close_points(pts, r=MERGE_VERT_DIST):
    """Greedy clustering; returns list of (x,y) floats."""
    pts = [tuple(map(float,p)) for p in pts]
    centers=[]
    for p in pts:
        found=False
        for i,c in enumerate(centers):
            if np.hypot(c[0]-p[0], c[1]-p[1]) <= r:
                centers[i] = ((c[0]+p[0])/2.0, (c[1]+p[1])/2.0)
                found=True; break
        if not found: centers.append(p)
    return centers

# ---------- graph extraction with correct boundary handling ----------
def extract_graph_with_boundary(gray):
    h,w = gray.shape
    rect, rect_corners = detect_rectangle_via_hough(gray)

    sk = skeletonize_image(gray)
    sk = prune_spurs(sk)

    # true vertices = junctions (deg>=3) + endpoints (deg==1) + rectangle corners
    verts=[]
    for y in range(h):
        for x in range(w):
            if sk[y,x]!=1: continue
            d = len(neighbors8(y,x,sk))
            if d==1 or d>=3:
                verts.append((x,y))
    verts += [(cx,cy) for (cx,cy) in rect_corners]
    verts = merge_close_points(verts, r=MERGE_VERT_DIST)

    # integer map for quick lookups
    vmap = { (int(round(v[1])), int(round(v[0]))): i for i,v in enumerate(verts) }

    def is_vertex_pix(y,x):
        return (y,x) in vmap

    edges=set()
    visited=set()

    # walk skeleton from each vertex to the next vertex; if chain runs off, project to rectangle & create a boundary vertex
    for i,(vx,vy) in enumerate(verts):
        y0, x0 = int(round(vy)), int(round(vx))
        starts = [(y0,x0)] + [(y0+dy, x0+dx) for dy,dx in NB8]
        starts = [(yy,xx) for (yy,xx) in starts if 0<=yy<h and 0<=xx<w and sk[yy,xx]==1]
        for sy,sx in starts:
            for ny,nx in neighbors8(sy,sx, sk):
                key=(sy,sx,ny,nx)
                if key in visited: continue
                path=[(sy,sx),(ny,nx)]
                py,px=sy,sx
                cy,cx=ny,nx
                while True:
                    nbs = neighbors8(cy,cx, sk)
                    nbs = [p for p in nbs if p!=(py,px)]
                    if not nbs:  # dead-end -> project last segment to rectangle
                        ip = project_to_rectangle((float(px),float(py)), (float(cx),float(cy)), rect)
                        if ip is not None:
                            verts.append(ip)
                            # rebuild map lazily
                            vmap.clear()
                            for k,(xx,yy) in enumerate(verts):
                                vmap[(int(round(yy)), int(round(xx)))] = k
                            a = i
                            b = vmap[(int(round(ip[1])), int(round(ip[0])))]
                            if a!=b: edges.add(tuple(sorted((a,b))))
                        break
                    if len(nbs)>=2 or is_vertex_pix(cy,cx):
                        a = i
                        b = vmap.get((cy,cx))
                        if b is None:  # snap to nearest
                            ds = [np.hypot(xx-cx, yy-cy) for (xx,yy) in verts]
                            b = int(np.argmin(ds))
                        if a!=b: edges.add(tuple(sorted((a,b))))
                        break
                    py,px = cy,cx
                    cy,cx = nbs[0]
                visited.add(key)

    # ensure rectangle edges are present
    corner_ids = [vmap[(int(round(cy)),int(round(cx)))] for (cx,cy) in rect_corners]
    for e in [(corner_ids[0],corner_ids[1]),
              (corner_ids[1],corner_ids[2]),
              (corner_ids[2],corner_ids[3]),
              (corner_ids[3],corner_ids[0])]:
        edges.add(tuple(sorted(e)))

    return list(verts), [list(e) for e in sorted(edges)]

# ---------- face extraction (cells) ----------
def polygon_area(coords):
    x = coords[:,0]; y=coords[:,1]
    return 0.5*np.sum(x*np.roll(y,-1) - np.roll(x,-1)*y)

def polygon_centroid(coords):
    x = coords[:,0]; y=coords[:,1]
    a = np.sum(x*np.roll(y,-1) - np.roll(x,-1)*y)
    if abs(a) < 1e-9:
        return np.array([x.mean(), y.mean()])
    cx = np.sum((x+np.roll(x,-1))*(x*np.roll(y,-1)-np.roll(x,-1)*y))/(3*a)
    cy = np.sum((y+np.roll(y,-1))*(x*np.roll(y,-1)-np.roll(x,-1)*y))/(3*a)
    return np.array([cx,cy])

def build_cells(vertices, edges, img_shape):
    """Planar face walk on straight-edge graph -> bounded faces (cells)."""
    V = np.array(vertices, dtype=float)
    h,w = img_shape

    # adjacency with angles
    adj = {i:set() for i in range(len(V))}
    for a,b in edges:
        adj[a].add(b); adj[b].add(a)
    ang = {}
    for i in range(len(V)):
        vi = V[i]
        ang[i] = {}
        for j in adj[i]:
            vj = V[j]
            ang[i][j] = atan2(vj[1]-vi[1], vj[0]-vi[0])
    # neighbors sorted CCW
    nbr_sorted = {i: sorted(list(adj[i]), key=lambda j: ang[i][j]) for i in adj}

    # half-edge visitation
    visited_he = set()
    faces = []          # list of ordered vertex indices per face (closed loop)
    he_start_for_face = []

    # iterate all half-edges
    for a,b in edges:
        for (u,v) in [(a,b),(b,a)]:
            if (u,v) in visited_he: continue
            face = []
            start = (u,v)
            cur_u, cur_v = u, v
            while True:
                visited_he.add((cur_u,cur_v))
                face.append(cur_u)
                # at cur_v, rotate "right" (clockwise) to keep face on left of half-edge
                nbrs = nbr_sorted[cur_v]
                if cur_u not in nbrs: break  # broken adjacency (shouldn't happen)
                k = nbrs.index(cur_u)
                nxt = nbrs[(k-1) % len(nbrs)]  # previous in CCW list = clockwise
                cur_u, cur_v = cur_v, nxt
                if (cur_u,cur_v) == start:
                    break
            if len(face)>=3:
                faces.append(face)
                he_start_for_face.append(start)

    # dedupe faces by vertex cycles (normalize rotation)
    def norm_cycle(cyc):
        m = min(range(len(cyc)), key=lambda i: cyc[i])
        c1 = cyc[m:]+cyc[:m]
        c2 = c1[:1]+list(reversed(c1[1:]))
        return tuple(min(c1,c2))
    uniq = {}
    for f in faces:
        key = norm_cycle(f)
        uniq[key]=f
    faces = list(uniq.values())

    # remove outer face (largest absolute area)
    areas = [abs(polygon_area(V[np.array(f)])) for f in faces]
    if areas:
        outer_idx = int(np.argmax(areas))
        faces.pop(outer_idx)

    # build sub_dict_* structure + neighbors via shared edges
    edge_to_faces = {}
    for fid,f in enumerate(faces):
        for i in range(len(f)):
            a=f[i]; b=f[(i+1)%len(f)]
            e = tuple(sorted((a,b)))
            edge_to_faces.setdefault(e, []).append(fid)

    sub = {}
    for fid,f in enumerate(faces):
        poly = V[np.array(f)]
        # seed = polygon centroid, normalized to [0,1]
        c = polygon_centroid(poly)
        seed_norm = [float(c[0]/(w-1)), float(c[1]/(h-1))]
        # edges around this face
        e_local = [[int(f[i]), int(f[(i+1)%len(f)])] for i in range(len(f))]
        # neighbors = faces sharing an edge
        neigh=set()
        for e in e_local:
            other = edge_to_faces.get(tuple(sorted(e)), [])
            for ff in other:
                if ff!=fid: neigh.add(ff)
        sub[f"sub_dict_{fid}"] = {
            "seed number": fid,
            "seed coordinates": seed_norm,
            "neighbors": sorted(list(neigh)),
            "edges": e_local,
            "cell vertices": [int(v) for v in f]
        }
    return sub

# ---------- user-facing pipeline ----------
def image_to_explicit_voronoi_dict(image_path, plot=True):
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # graph (vertices + edges) directly from the drawing
    vertices, edges = extract_graph_with_boundary(gray)

    # faces -> your exact sub_dict_* format
    explicit = build_cells(vertices, edges, gray.shape)

    if plot:
        # quick visualization
        fig, ax = plt.subplots(figsize=(8,6))
        # ax.imshow(gray, cmap='gray')
        V = np.array(vertices, float)
        # ax.plot(V[:,0], V[:,1], 'ro', ms=5, label='Vertices')
        # for a,b in edges:
        #     ax.plot([V[a,0],V[b,0]], [V[a,1],V[b,1]], 'g-', lw=2)
        # ax.set_title("Reconstructed Explicit Voronoi")
        # ax.legend(loc='lower left')
        # plt.show()

    return explicit, vertices, edges

# --------------- example ---------------
if __name__ == "__main__":
    path = r"C:\Users\user\OneDrive\PHD\Repositories\Inverse_Voronoi\Image Processing\voronoi_image.png"
    explicit_voronoi, vertices, edges = image_to_explicit_voronoi_dict(path, plot=True)

    # show one cell example and count
    print(f"#cells: {len(explicit_voronoi)}  #verts: {len(vertices)}  #edges: {len(edges)}")
    first_key = list(explicit_voronoi.keys())[0]
    print(explicit_voronoi)

    # extract edges from explicit voronoi
    extracted_edges = []
    for cell in explicit_voronoi:
        for edges in explicit_voronoi[cell]['edges']:
            if edges not in extracted_edges and edges[::-1] not in extracted_edges:
                extracted_edges.append(edges)

    print('Extracted Edges:', extracted_edges)
    # extract vertices from explicit voronoi
    extracted_vertices = []
    for cell in explicit_voronoi:
        for vertex in explicit_voronoi[cell]['cell vertices']:
            if vertex not in extracted_vertices:
                extracted_vertices.append(vertex)
    print('Extracted Vertices:', extracted_vertices)

    import matplotlib.pyplot as plt
    # visualize extracted edges and vertices
    for edge in extracted_edges:
        pt1 = vertices[edge[0]]
        pt2 = vertices[edge[1]]
        plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'b-')
    plt.show()
