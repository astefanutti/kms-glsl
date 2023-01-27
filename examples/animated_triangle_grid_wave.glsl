// Copied from https://www.shadertoy.com/view/7sycz3
// Created by https://www.shadertoy.com/user/Shane

/*

    Animated Triangle Grid Weave
    ----------------------------

    I did this a while ago in preparation for an icosahedral example I
    was working on. Icosahedrons consist of triangles, so if you can
    get a random animated weave to work on a flat triangle grid, it
    should work on the surface of any equilateral triangle based entity.

    Anyway, the weave pattern itself is pretty straight forward:
    Construct a triangle grid, create a separate arc around each
    triangle cell vertex, then render each of them in random order --
    A quick way to do that is to randomly rotate the local triangle
    cell coordinates then render the arcs in the same order.

    The animation didn't present too many problems, but it took me a
    while to realize that I'd need double arcs containing separate
    paths running in opposite directions for things to work on a random
    weave -- The reasoning relates to how animated hexagon Truchets
    work... It's not important why.

    The design was rushed, but it's clean looking. I got used to the
    original flat look and template colors, so left them as is. I'll
    post the polyhedral example at some stage.


    Other examples:

    // A different kind of weave produced in a 3D chainlink style. Very cool.
    tri truch tralala - flockaroo
    https://www.shadertoy.com/view/WlS3WV

    // An extruded simplex weave, but with no animation. It takes a while to
    // compile, but runs well enough... I'll make it compile faster later.
    Simplex Weave - Shane
    https://www.shadertoy.com/view/WdlSWl

*/


// Standard 2D rotation formula.
mat2 rot2(in float a){ float c = cos(a), s = sin(a); return mat2(c, -s, s, c); }


// IQ's vec2 to float hash.
float hash21(vec2 p){  return fract(sin(dot(p, vec2(27.619, 57.583)))*43758.5453); }


////////
// A 2D triangle partitioning. I've dropped in an old routine here.
// It works fine, but could do with some fine tuning. By the way, this
// will partition all repeat grid triangles, not just equilateral ones.

// Skewing coordinates. "s" contains the X and Y skew factors.
vec2 skewXY(vec2 p, vec2 s){ return mat2(1, -s.yx, 1)*p; }

// Unskewing coordinates. "s" contains the X and Y skew factors.
vec2 unskewXY(vec2 p, vec2 s){ return inverse(mat2(1, -s.yx, 1))*p; }

// Triangle scale: Smaller numbers mean smaller triangles, oddly enough. :)
const float scale = 1./5.;

// Rectangle scale.
const vec2 rect = (vec2(1./.8660254, 1))*scale;
// Skewing half way along X, and not skewing in the Y direction.
const vec2 sk = vec2(rect.x*.5, 0)/scale; // 12 x .2


float gTri;
vec4 getTriVerts(vec2 p, inout vec2[3] vID, inout vec2[3] v){

    // Skew the XY plane coordinates.
    p = skewXY(p, sk);

    // Unique position-based ID for each cell. Technically, to get the central position
    // back, you'd need to multiply this by the "rect" variable, but it's kept this way
    // to keep the calculations easier. It's worth putting some simple numbers into the
    // "rect" variable to convince yourself that the following makes sense.
    vec2 id = floor(p/rect) + .5;
    // Local grid cell coordinates -- Range: [-rect/2., rect/2.].
    p -= id*rect;

    // Equivalent to:
    //gTri = p.x/rect.x < -p.y/rect.y? 1. : -1.;
    // Base on the bottom (-1.) or upside down (1.);
    gTri = dot(p, 1./rect)<0.? 1. : -1.;

    // Puting the skewed coordinates back into unskewed form.
    p = unskewXY(p, sk);

    // Vertex IDs for the quad.
    const vec2[4] vertID = vec2[4](vec2(-.5, .5), vec2(.5), vec2(.5, -.5), vec2(-.5));

    // Vertex IDs for each partitioned triangle.
    if(gTri<0.){
        vID = vec2[3](vertID[0], vertID[2], vertID[1]);
    }
    else {
        vID = vec2[3](vertID[2], vertID[0], vertID[3]);
    }

    // Triangle vertex points.
    for(int i = 0; i<3; i++) v[i] = unskewXY(vID[i]*rect, sk); // Unskew.

    // Centering at the zero point.
    vec2 ctr = v[2]/3.; // Equilateral equivalent to: (v[0] + v[1] + v[2])/3.;
    p -= ctr;
    v[0] -= ctr;
    v[1] -= ctr;
    v[2] -= ctr;

    // Specific centered triangle ID.
    ctr = vID[2]/3.; //(vID[0] + vID[1] + vID[2])/3.;//vID[2]/2.;
    id += ctr;
    // Not used here, but for jigsaw pattern creation, etc, the vertex IDs
    // need to be correctly centered too.
    //vID[0] -= ctr; vID[1] -= ctr; vID[2] -= ctr;


    // Triangle local coordinates (centered at the zero point) and
    // the central position point (which acts as a unique identifier).
    return vec4(p, id);
}

//////////
// Rendering a colored distance field onto a background. I'd argue that
// this one simple function is the key to rendering most vector styled
// 2D Photoshop effects onto a canvas. I've explained it in more detail
// before. Here are the key components:
//
// bg: background color, fg: foreground color, sf: smoothing factor,
// d: 2D distance field value, tr: transparency (0 - 1).
vec3 blend(vec3 bg, vec3 fg, float sf, float d, float tr){
    return mix(bg, fg, (1. - smoothstep(0., sf, d))*tr);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord){
    // Aspect correct screen coordinates.
    vec2 uv = (fragCoord - iResolution.xy*.5)/iResolution.y;

    // Global scaling and translation.
    float gSc = 1.;
    // Smoothing factor, based on global scaling.
    float sf = 1./iResolution.y*gSc;
    // Depending on perspective; Moving the oject toward the bottom left,
    // or the camera in the north easterly (top right) direction.
    vec2 p = uv*gSc - vec2(-.57735, -1)*iTime/50.;

    // Cell coordinate, ID and triangle orientation id.
    // Cell vertices and vertex ID.
    vec2[3] v, vID;

    // Returns the local coordinates (centered on zero), cellID, the
    // triangle vertex ID and relative coordinates.
    vec4 p4 = getTriVerts(p, vID, v);
    p = p4.xy;
    vec2 triID = p4.zw;
    float tri = gTri;

    // Grid triangles. Some are upside down.
    //vec2 q = tri<0.? p*vec2(1, -1) : p;
    vec2 q = p*vec2(1, tri); // Equivalent to the line above.
    float tr = (max(abs(q.x)*.8660254 + q.y*.5, -q.y) - scale/3.);

    // Object direction, based on triangle ID.
    vec3 dir = tri<0.? vec3(-1, 1, 1) : vec3(1, -1, -1);

    // Nearest vertex ID.
    float vert = 1e5;
    vec3 arc, ang;
    float sL = length(v[0] - v[1]);

    // Random value based on the overall triangle ID.
    float rnd = hash21(triID + .1);

    // Random rotation, in incrents of 120 degrees to maintain symmetry.
    p = rot2(floor(rnd*36.)*6.2831/3.)*p;

    // Nearest vertex, vertex-arc and angle (subtended from each vertex) calculations.
    vec2 vertID;
    for(int i = 0; i<3; i++){
        float vDist = length(p - v[i]);
        if(vDist<vert){
            vert = vDist; // Nearest vertex.
            vertID = triID + vID[i]; // Nearest vertex ID.
        }

        // One of three arcs that loop around each vertex. This is still
        // circle distance at this point, but the rest of the calculations
        // are performed outside the loop (see below).
        arc[i] = (vDist - sL/2.);

        // Angle of each pixel on each arc. As above, further calculations
        // are performed outside the loop for speed.
        vec2 vp = p - v[i];
        ang[i] = atan(vp.y, vp.x);
    }

    // Number of rotating entities on each arc. Due to triangle symmetry, it needs
    // to be a multiple of 6, and with the two arc setup here, multiples of 12...
    // Maybe others will work, but the double arc setup complicates things a bit.
    vec3 aNum = vec3(12);
    vec3 aNum0 = aNum; // Using a copy of the initial number to scale values later.

    // Lane direction.
    vec3 laneDir = dir;
    // Reverse the direction of inside lanes and halve the number of objects -- Inside
    // lanes cover less distance, so fewer objects space out better. By the way, there's
    // probably a cleverer way to write the following, but my mind's feeling lazy. :)
    if(arc.x<0.){ laneDir.x *= -1.; aNum.x /= 2.; }
    if(arc.y<0.){ laneDir.y *= -1.; aNum.y /= 2.; }
    if(arc.z<0.){ laneDir.z *= -1.; aNum.z /= 2.; }
    arc = abs(arc); // Turning the circle into an arc.
    arc = abs(arc - .04); // Doubling the arcs. One inside and the other outside.
    arc -= .036; // Arc thickness.
    //laneDir = vec3(1);

    // The final figure, "aNum0", is a smoothing factor fudge. Since we're repeating by
    // a factor of "aNum", that number needs to be compensated for... There's probably
    // an exact way to do it (derivatives spring to mind), but this will do.
    vec3 ani = (abs(fract(ang/6.2831*aNum + iTime*dir*laneDir*.5) - .5)*2. - .66)/aNum0/2.;
    ang = max(arc + .025, ani);
    //vec3 ang2 = max(arc + .026, (abs(ani + .25/aNum0/2.) - .2/aNum0/2.));

    // Background, set to black.
    vec3 col = vec3(0);

    // Rendering some green triangles onto the background, but leaving the edges.
    col = blend(col, vec3(.7, 1, .4), sf, tr + .0035, 1.);

    // Triangle grid vertices.
    vert -= .0225; // Vertex radius.
    col = blend(col, vec3(0), sf, vert, 1.);
    col = blend(col, vec3(.8, 1, .2), sf, vert + .005, 1.);
    col = blend(col, vec3(0), sf, vert + .018, 1.);

    // Resolution factor for shadow width -- It's a hack to make sure shadows
    // have the same area influence at different resolutions. If you think it's
    // confusing, you'll get no arguments from me. :)
    float resF = iResolution.y/450.;

    // Rendering the three sets of double arcs.
    for(int i = 0; i<3; i++){

        // Arcs: Rails, edges, etc.
        col = blend(col, vec3(0), sf*8.*resF, arc[i], .5);
        col = blend(col, vec3(0), sf, arc[i], 1.);

        col = blend(col, vec3(1, .9, .85), sf, arc[i] + .005, 1.);
        col = blend(col, vec3(0), sf, arc[i] + .016, 1.);
        col = blend(col, vec3(1, .15, .3), sf, arc[i] + .016 + .005, 1.);
        col = blend(col, vec3(0), sf*4.*resF, abs(arc[i] + .016 + .005), .25);

        // The animated strips.
        col = blend(col, vec3(0), sf*4.*resF, ang[i], .25);
        col = blend(col, vec3(0), sf, ang[i], 1.);
        col = blend(col, vec3(1, .9, .3), sf, ang[i] + .005, 1.);

        // Hood and trunk, for the overhead cars look.
        // "ang2" needs uncommenting for it to work
        //col = blend(col, vec3(0), sf, ang2[i], .65);
        //col = blend(col, vec3(1, .9, .3)*.7, sf, ang2[i] + .005, 1.);

    }

    // Subtle vignette.
    uv = fragCoord/iResolution.xy;
    col *= pow(16.*uv.x*uv.y*(1. - uv.x)*(1. - uv.y) , 1./32.);
    // Colored variation.
    //col = mix(col.zyx, col, pow(16.*uv.x*uv.y*(1. - uv.x)*(1. - uv.y) , 1./16.));

    // Rough gamma correction.
    fragColor = vec4(sqrt(max(col, 0.)), 1);
}
