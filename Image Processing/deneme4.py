import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

# ------------------- knobs you can tweak -------------------
THRESH_BIN = 200          # binarization threshold
NOISE_MIN_BRANCH = 3      # prune spur branches shorter than this (px)
NEAR_BORDER_PX = 4        # how close to border counts as "on" the rectangle
MERGE_VERT_DIST = 3       # merge clustered vertices (px)
# -----------------------------------------------------------

def to_binary_inv(gray):
    _, th = cv2.threshold(gray, THRESH_BIN, 255, cv2.THRESH_BINARY_INV)
    return (th // 255).astype(np.uint8)  # 0/1

def skeletonize_image(gray):
    bin01 = to_binary_inv(gray)
    skel = skeletonize(bin01).astype(np.uint8)  # 0/1
    return skel

# --- utilities on pixel graph ---
NB8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
def neighbors8(y,x, sk):
    h,w = sk.shape
    out=[]
    for dy,dx in NB8:
        yy,xx = y+dy, x+dx
        if 0<=yy<h and 0<=xx<w and sk[yy,xx]==1:
            out.append((yy,xx))
    return out

def degree(y,x, sk):
    return len(neighbors8(y,x, sk))

def prune_spurs(skel, keep_border=True):
    """Remove tiny spurs (degree-1 chains) shorter than NOISE_MIN_BRANCH."""
    sk = skel.copy()
    h,w = sk.shape
    changed = True
    while changed:
        changed=False
        to_del=[]
        for y in range(h):
            for x in range(w):
                if sk[y,x]==1 and degree(y,x,sk)==1:
                    if keep_border and (x==0 or y==0 or x==w-1 or y==h-1):
                        continue
                    # walk ahead and count length
                    path=[(y,x)]
                    py,px = y,x
                    ny,nx = neighbors8(y,x,sk)[0]
                    while sk[ny,nx]==1 and degree(ny,nx,sk)==2 and len(path)<NOISE_MIN_BRANCH+1:
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
    """Detect the outer rectangle: return (xL,yT,xR,yB) as floats and 4 corner points."""
    h,w = gray.shape
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80,
                            minLineLength=int(0.6*min(h,w)), maxLineGap=10)
    if lines is None:
        # fall back to tight image border
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
    """Return intersection point of segments p1p2 & q1q2 (as infinite lines)."""
    x1,y1 = p1; x2,y2=p2; x3,y3=q1; x4,y4=q2
    den = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(den) < 1e-9: return None
    px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4))/den
    py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4))/den
    return (px,py)

def project_to_rectangle(p_prev, p_end, rect):
    """Intersect ray (p_prev->p_end) with rect; return intersection point."""
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
    """Simple clustering by greedy merging."""
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

def extract_graph_with_boundary(gray):
    """
    Skeleton-walk graph extraction that (a) creates true junctions/endpoints,
    (b) snaps boundary hits to precise rectangle intersections,
    (c) returns vertices + edges.
    """
    h,w = gray.shape
    rect, rect_corners = detect_rectangle_via_hough(gray)

    sk = skeletonize_image(gray)
    sk = prune_spurs(sk)

    # ---- find junctions/endpoints on the skeleton ----
    vertex_pix=[]
    for y in range(h):
        for x in range(w):
            if sk[y,x] != 1: continue
            d = degree(y,x, sk)
            if d==1 or d>=3:
                vertex_pix.append((x,y))
    # also add exact rectangle corners
    vertex_pix += [(cx,cy) for (cx,cy) in rect_corners]

    # merge clusters
    vertex_pix = merge_close_points(vertex_pix, r=MERGE_VERT_DIST)
    vertex_map = { (int(round(y)),int(round(x))) : i
                   for i,(x,y) in enumerate(vertex_pix) }

    def is_vertex(y,x):
        return (y,x) in vertex_map

    # ---- walk edges along skeleton ----
    visited=set()
    edges=[]
    for (x0,y0) in vertex_pix:
        y0i, x0i = int(round(y0)), int(round(x0))
        # if starting pixel is not exactly on the skeleton, look around 1px
        start_nbs = [(y0i,x0i)]
        for dy,dx in NB8: start_nbs.append((y0i+dy, x0i+dx))
        start_nbs = [(yy,xx) for (yy,xx) in start_nbs
                     if 0<=yy<h and 0<=xx<w and sk[yy,xx]==1]
        for sy,sx in start_nbs:
            for ny,nx in neighbors8(sy,sx, sk):
                key = (sy,sx,ny,nx)
                if key in visited: 
                    continue
                path=[(sy,sx),(ny,nx)]
                py,px=sy,sx
                cy,cx=ny,nx
                while True:
                    nbs = neighbors8(cy,cx, sk)
                    nbs = [p for p in nbs if p!=(py,px)]
                    if not nbs:
                        # dead end: project last segment to rectangle
                        p_prev = (float(px), float(py))
                        p_end  = (float(cx), float(cy))
                        ip = project_to_rectangle(p_prev, p_end, rect)
                        if ip is not None:
                            # add/merge this intersection as a vertex
                            vertex_pix.append(ip)
                            vertex_pix_merged = merge_close_points(vertex_pix, r=MERGE_VERT_DIST)
                            # rebuild map (rare path; keeps code simple)
                            vertex_pix[:] = vertex_pix_merged
                            vertex_map.clear()
                            for i,(vx,vy) in enumerate(vertex_pix):
                                vertex_map[(int(round(vy)),int(round(vx)))] = i
                            v1 = vertex_map[(int(round(y0)),int(round(x0)))]
                            v2 = vertex_map[(int(round(ip[1])), int(round(ip[0])))]
                            if v1!=v2 and [v1,v2] not in edges and [v2,v1] not in edges:
                                edges.append([v1,v2])
                        break
                    if len(nbs)>=2 or is_vertex(cy,cx):
                        # hit a junction/vertex
                        v1 = vertex_map[(int(round(y0)),int(round(x0)))]
                        v2 = vertex_map.get((cy,cx))
                        if v2 is None:
                            # snap to nearest existing vertex
                            dists=[(i, np.hypot(vx-cx, vy-cy))
                                   for i,(vx,vy) in enumerate([(vx,vy) for (vx,vy) in vertex_pix])]
                            v2 = min(dists, key=lambda z:z[1])[0]
                        if v1!=v2 and [v1,v2] not in edges and [v2,v1] not in edges:
                            edges.append([v1,v2])
                        break
                    # continue straight
                    py,px = cy,cx
                    cy,cx = nbs[0]
                visited.add(key)

    # finally, ensure rectangle sides are present as edges
    # link rectangle corners in order
    corner_ids = []
    for (cx,cy) in rect_corners:
        corner_ids.append(vertex_map[(int(round(cy)),int(round(cx)))])
    rect_edges = [[corner_ids[0],corner_ids[1]],
                  [corner_ids[1],corner_ids[2]],
                  [corner_ids[2],corner_ids[3]],
                  [corner_ids[3],corner_ids[0]]]
    for e in rect_edges:
        if e not in edges and e[::-1] not in edges:
            edges.append(e)

    return vertex_pix, edges, rect

def build_explicit_voronoi(gray):
    vertices, edges, rect = extract_graph_with_boundary(gray)
    explicit = {
        "vertices": [[float(x), float(y)] for (x,y) in vertices],
        "edges": [[int(a), int(b)] for a,b in edges],
        # cells can be computed later if you need faces/topology
    }
    return explicit

def plot_explicit(explicit, img, title="Reconstructed Explicit Voronoi"):
    fig, ax = plt.subplots(figsize=(8,6))
    ax.imshow(img, cmap='gray')
    V = np.array(explicit["vertices"])
    if len(V):
        ax.plot(V[:,0], V[:,1], 'ro', ms=5, label='Vertices')
    for a,b in explicit["edges"]:
        ax.plot([V[a,0],V[b,0]], [V[a,1],V[b,1]], 'g-', lw=2)
    ax.set_title(title)
    ax.legend(loc='lower left')
    plt.show()

# -------------------- run --------------------
if __name__ == "__main__":
    path = r"C:\Users\user\OneDrive\PHD\Repositories\Inverse_Voronoi\Image Processing\voronoi_image.png"
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)

    explicit_voronoi = build_explicit_voronoi(img)
    print("Vertices:", len(explicit_voronoi["vertices"]))
    print("Edges:", len(explicit_voronoi["edges"]))
    plot_explicit(explicit_voronoi, img)
    print(explicit_voronoi)
