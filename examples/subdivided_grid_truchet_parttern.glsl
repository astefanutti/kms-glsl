// Copied from https://www.shadertoy.com/view/NdKfRD
// Created by https://www.shadertoy.com/user/Shane

/*

    Subdivided Grid Truchet Pattern
    -------------------------------

    Rendering a Truchet pattern onto a randomly subdivided square grid. It can
    also be referred to as a multiscale Truchet pattern. This particular
    variation is pretty common, so there's a better than even chance that
    you've seen some around. I put together a multiscaled Truchet example a
    while ago, and this was one of many variations I tried out. I was saving it
    for a pseudo 3D version, but have procrastinated on it for too long, so
    figured I'd repackage the original and post it in 2D form.

    If it's not immediately obvious from the pattern, it's a simple rendering
    of concentric-circle Truchet rings onto a randomly subdivided grid.
    Uncommenting the "SHOW_GRID" define should make it more clear. When you
    subdivide down a level, you simply render half the amount of concentric
    rings in the next cell, and so forth. Common sense dictates that you should
    choose an initial number of rings that can be halved to maintain integer
    rings for the next subdivision; For instance, 12 rings in the big cells
    will allow for 2 more sudivisions (12-6-3).

    This particular setup subdivides cells without considering neighbors, so
    is shorter and easier to understand, but doesn't allow for rendering things
    like long shadows, etc. There's a tiny bit of extra dressup code here, but
    this is a pretty basic example with a reasonably small code footprint. A
    simple black and white pattern would require far less code still... I'll
    leave that as an exercise for the code golfing crowd. :)

    I would eventually like to produce a more 3D looking version. It's also
    possible to produce these on other kinds of subdivision grids, like
    triangles and others -- I have a triangle version floating around somewhere,
    and a pattern mapped onto a 3D surface that I'd like to post later.



    Based on:

    A multiscale Truchet pattern that considers neighboring cells.
    Quadtree Truchet - Shane
    https://www.shadertoy.com/view/4t3BW4

    // A concentric circle Truchet pattern in under a tweet. The code
    // contains fewer characters than it took to write this description. :)
    70s Wallpaper - Shane
    https://www.shadertoy.com/view/ls33DN


*/


//////////////

// Stripe color - Gold: 0, Pink: 1, Green: 2, Silver: 3.
#define COLOR 0

// Background stripe shade - Black: 0, White: 1.
#define COLORB 1

// Show the randomly subdivided grid.
//#define SHOW_GRID

//////////////


// Standard 2D rotation formula.
mat2 rot2(in float a){ float c = cos(a), s = sin(a); return mat2(c, -s, s, c); }


// IQ's vec2 to float hash.
float hash21(vec2 p){  return fract(sin(dot(p, vec2(27.619, 57.583)))*43758.5453); }


// Global scale -- It's pretty lazy putting it here.
float gSc = 0.;


// The wall and floor pattern, which is just something quick and effective.
// It's an offset row square grid pattern with some random subdivision.
vec4 distField(vec2 p, float sc){


    vec2 q = p;
    // Partitioning into cells and providing the local cell ID
    // and local coordinates.
    // Offset alternate rows: I've left this out of the options, but it's
    // possible to render some Truchet patterns on offset grids, which can
    // produce interesting results.
    //if(mod(floor(p.y/sc), 2.)<.5) p.x += sc/2.;
    // Cell ID and local coordinates.
    vec2 ip = floor(p/sc);
    p -= (ip + .5)*sc;

    for(int i = 0; i<2; i++){
        // Random subdivision -- One big cell becomes four smaller ones.
        if(hash21(ip + float(i + 1)*.007)<.5){//(1./float(i + 2))
            sc /= 2.; // Cut the scale in half.
            p = q;
            // New cell ID and local coordinates.
            ip = floor(p/sc);
            p -= (ip + .5)*sc;

        }
    }

    // Global scale copy.
    gSc = sc;

    // Returning the local coordinates and local cell ID. Note that the
    // distance has been rescaled by the scaling factor.
    return vec4(p, ip);
}


void mainImage( out vec4 fragColor, in vec2 fragCoord ){

    // Aspect corret coordinates.
    vec2 uv = (fragCoord - iResolution.xy*.5)/iResolution.y;

    // Smoothing factor.
    float sf = 1.5/iResolution.y;

    // Canvas coordinates with translation.
    vec2 p = uv + vec2(1, .5)*iTime/16.;

    // Scene field calculations.


    // Randomly subdivided grid object.
    float sc = iResolution.y>360.01? 1./4. : 1./4.*450./360.; // Canvas size scaling.
    vec4 d4 = distField(p, sc);
    //
    // Grid cell's local coordinates and unique ID.
    vec2 q = d4.xy;
    vec2 iq = d4.zw;

    // Randomly rotating the square cell's local coordinates.
    float rnd = hash21(iq*gSc + .23);
    q = rot2(floor(rnd*36.)*6.2831/4.)*q;

    // Lighting.
    vec2 ld = normalize(vec2(1, -2));
    ld = rot2(floor(rnd*36.)*6.2831/4.)*ld; // Random rotation to match.

    // Grid cell square -- Used to show the grid cell outlines.
    float bord = max(abs(q.x),abs(q.y)) -  gSc/2.;


    // Concentric circle line number for the larger cells. Factors of
    // four will work. For more subdivisions, larger factors of four
    // are required.
    const float lNum = 2.*4.;
    float lW = sc/lNum; // Concentric line width.
    float ew = lW/5.; // Edge width.


    // Truchet circle distances from each diagonal.
    vec2 tr = vec2(length(q - gSc/2.), length(q + gSc/2.));
    // Offset reading for highlighting purposes.
    vec2 tr2 = vec2(length(q - gSc/2. - ld*.002), length(q + gSc/2. - ld*.002));


    vec2 ln = abs(mod(tr, lW) - .5*lW) - .25*lW - ew/2.;
    vec2 ln2 = abs(mod(tr2, lW) - .5*lW) - .25*lW - ew/2.;

    // Highlighting or bump factor.
    vec2 b = max(smoothstep(0., .1, ln2 - ln), 0.)/.002;
    b *= b*4.5; // Tweaking.

    // Truchet arcs from the mid edges.
    //tr = abs(tr - gSc/2.) - gScx/2. + lW/4. - ew/2.;
    // Truchet circles centered on the diagonals of side length radius.
    tr = tr - gSc + lW/4. - ew/2.;
    // Offset reading for highlighting purposes.
    tr2 = tr2 - gSc + lW/4. - ew/2.;

    // Background stripe color.
    #if COLORB == 0
    vec3 lnColB = vec3(.1); // Black.
    #else
    vec3 lnColB = vec3(1); // White.
    #endif

    // Rendering onto the background.
    //float pat = dot(sin(uv*4. - cos(uv.yx*8.)), vec2(.15)) + .5;
    // Fake height highlighting pattern. There's an accurate way to represent height,
    // (angles to edges, etc) and this isn't it, but this is will do for now.
    float pat = .8 - smoothstep(.0, 1., length(q)/(gSc/2.))*.5;
    #if COLOR == 0
    vec3 lnCol = mix(vec3(.65, .2, .06), vec3(1, .45, .2), pat); // Gold.
    #elif COLOR == 1
    vec3 lnCol = mix(vec3(.6, .3, .9), vec3(1, .1, .3), pat); // Pink.
    #elif COLOR == 2
    vec3 lnCol = mix(vec3(.25, .6, .1), vec3(.15, .4, .5), pat); // Green.
    #else
    vec3 lnCol = mix(vec3(.3), vec3(.45, .55, .7), pat); // Siver.
    #endif

    // Scene color -- Set to the background.
    vec3 col = lnColB;

    // Rendering the two overlapping diagonally centered Truchet circles.
    for(int i = 0; i<2; i++){
        // Putting the concentric lines onto the arc.
        vec3 tCol = mix(lnColB, vec3(0), 1. - smoothstep(0., sf, ln[i]));
        tCol = mix(tCol, lnCol, 1. - smoothstep(0., sf, ln[i] + ew));

        // The arc itself; Drop shadow followed by coloring.
        col = mix(col, vec3(0), (1. - smoothstep(0., sf*iResolution.y/450., tr2[i]))*.9);
        col = mix(col, tCol*(.7 + b[i]), 1. - smoothstep(0., sf, tr[i] + ew/5.));

    }

    #ifdef SHOW_GRID
    // Show the grid.
    col = mix(col, vec3(0), (1. - smoothstep(0., sf*4., abs(bord) - .005))*.5);
    col = mix(col, vec3(0), 1. - smoothstep(0., sf, abs(bord) - .005));
    col = mix(col, vec3(1), 1. - smoothstep(0., sf, abs(bord) - .0005));
    #endif

    // Output to screen
    fragColor = vec4(sqrt(max(col, 0.)), 1);
}
