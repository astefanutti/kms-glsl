// Copied from https://www.shadertoy.com/view/NlKfzV
// Created by https://www.shadertoy.com/user/Shane

/*
    Hilbert Curve Animation
    -----------------------

    Rendering a Hilbert curve is one of those graphics programming cliches that
    I'm fond of, so I've been meaning to put one up for a while. Thanks to IQ's
    cool "Linear To Hilbert" example, Hilbert curves are back in vogue, so I
    figured it was as good a time as any to jump on the bandwagon. :) I have more
    interesting examples coming, but I wanted to start with something relatively
    simple.

    If you're confortable with subdividing and rotating space in a fractal fashion,
    then coding one won't be difficult. There are several different methods out
    there, and each have their merits. I needed to render a smoothly paramaterized
    curve inside a raymarching loop, which meant I had to at least attempt to use
    a quick routine.

    The method utilized here is very similar to the way in which Fabrice Neyret
    does it. The process essentially involves subdividing, splitting the resultant
    space into quadrants, making predetermined decisions for each quadrant, then
    repeating for the desired number of iterations. The decision making itself is
    pretty straight forward -- For example, if the coordinate is in the bottom
    right quadrant, rotate then flip along the X-axis, etc. I find the process is
    best explained in the article, "Hilbert Curve Coloring", which I've provided a
    link to below. I've provided a brief explanation in the code also.

    Most of the code is dressing up. The Hilbert curve procedure itself doesn't
    take up much room at all -- In fact, when I have time, I will put together a
    very simple version to accompany this. I have an extruded raymarched shader
    that I'll post pretty soon. After that, I'd like to post a proper 3D version.


    References and other examples:

    // In order to understand the basic Hilbert curve algorithm, this is my
    // preferred source.
    Tutorial: Hilbert Curve Coloring - Kerry Mitchell
    https://www.kerrymitchellart.com/tutorials/hilbert/hilbert-tutorial.html

    // Here is a simpler version using roughly the same algorith that I'm using.
    // Fabrice uses a different initial orientation that allows for more
    // streamlined code.
    Hilbert curve 5 - FabriceNeyret2
    https://www.shadertoy.com/view/XtjXW3

    // The idea to parameterize then animate the curve came from DjinnKahn's
    // example, below, which is well worth a look, if you haven't seen it already.
    // I was too lazy to read through the code, so I'm not sure how our methods
    // compare, but they do the same thing, so I'll assume they're similar. :)
    Sierpinski + Hilbert fractal - DjinnKahn
    https://www.shadertoy.com/view/ft3fR4

    // Dr2 has a few Hilbert curve related examples on here. This one is
    // his latest effort.
    Hilbertian Saltation - dr2
    https://www.shadertoy.com/view

    // Really fun to watch.
    Linear To Hilbert - iq
    https://www.shadertoy.com/view/llGcDm

*/

// The number of Hilbert curve iterations. I designed everything to work with the
// number 5. However, values 3 to 6 will look OK. Numbers outside that range
// haven't been accounted for.
const int iters = 5;

// Standard 2D rotation formula.
mat2 rot2(in float a){ float c = cos(a), s = sin(a); return mat2(c, -s, s, c); }

// IQ's vec2 to float hash.
float hash21(vec2 p){  return fract(sin(dot(p, vec2(27.619, 57.583)))*43758.5453); }

// IQ's unsigned line distance formula.
float distLine(vec2 p, vec2 a, vec2 b){
    p -= a; b -= a;
    float h = clamp(dot(p, b)/dot(b, b), 0., 1.);
    return length(p - b*h);
}

// Arc distance formula.
float dist(vec2 p){
    // Circular.
    return length(p);

    // Hard square edge.
    //p = abs(p);
    //return max(p.x, p.y);

    // Rounded square.
    //p = abs(p) - .015;
    //return min(max(p.x, p.y), 0.) + length(max(p, 0.)) + .015;

    // Diamond and octagon.
    //p = abs(p);
    //return abs(p.x + p.y)*.7; // Requires readjusting in the arc function.
    //return max(max(p.x, p.y), abs(p.x + p.y)*.7);
}

// A standard Hilbert curve routine with some extra parameterization hacked
// in at the end. It needs some tidying up, but it works pretty fast, all
// things considered. I've taken an approach that's very similar to Fabrice's
// example, here:
//
// Hilbert curve 5 - FabriceNeyret2
// https://www.shadertoy.com/view/XtjXW3

// Fabrice's started with a different orientation, which has led to slightly
// neater logic, which I might try to incorporate later.
vec4 hilbert(vec2 p){

    // Hacking in some scaling.
    float hSc = iResolution.y/iResolution.x*2.1;
    // If you scale the coordinates, you normally have to scale things back
    // after you've finished calculations.
    p *= hSc;

    // Saving the global coordinates prior to subdivision. I'm not experiencing
    // alignment glitches, but I'm using Fabrice's hack, just to be on the safe side. :)
    vec2 op = p + 1e-4;

    // Initial scale set to one.
    float sc = 1.;

    // Cell ordering vector -- Clockwise from the bottom left. If the new partitioned
    // frame is flipped, then this will be also.
    ivec4 valC = ivec4(0, 1, 2, 3);

    // Initate to top left quadrant cell.
    int val = 1;

    p = op; // Initialize.

    // Splitting the curve block into two. There's no real reason for doing this,
    // but I thought it filled the canvas dimensions a little better... Plus, I
    // like to complicate things for myself. :)
    if(p.x<0.) p.x = abs(p.x) - .5; // Left half -- Moved to the left.
    else { p.x = .5 - abs(p.x); p = -p.yx;  } // Right half -- Moved right and rotated CCW.

    p = fract(p + .5); // Needs to begin in the zero to one range.
    p *= vec2(-1, 1); // Not absolutely necessary, but we're forcing the top left quadrant split.

    // The horizontal and vertical vectors. I've adopted Fabrice's naming
    // convention (i and j), but have stored them in one vector.
    vec4 ij = vec4(1, 0, 0, 1), d12 = ij; //vec4(ij.xy, -ij.zw);

    int rn = 0; // Cell number.

    float dirX = 1.; // Hacked in to keep track of the left or right of the curve.

    for(int i = min(0, iFrame); i<iters; i++){
        // The quadrant splitting logic:
        // Bottom left: Rotate clockwise. Leave the first direction alone. The second points up.
        // Top left: Leave the space untouched. First direction points down. Second points right.
        // Top right: Flip across the X-axis. First direction points left. Second points down.
        // Bottom right: Rotate clockwise then flip across the X-axis. First points up. Second left alone.

        if(p.x>0.){
            // You need to reverse the rendering order of the two right cells.
            // In other words, swap(dir1, dir2);
            if(p.y>0.){ d12 = -ij; p.x = -p.x;  d12.xz = -d12.xz; val = 2; } // Top right.
            else { d12.xy = ij.zw;  /*d12.xy = ij.zw;*/
                p = p.yx*vec2(1, -1); d12 = d12.yxwz*vec4(1, -1, 1, -1);
                dirX *= -1.; val = 3; // Bottom right  (Exit).
            }

            // Flip vector directions on the right -- You could incorporate this into the
            // lines above, if you wanted to.
            d12 = d12.zwxy;

            valC = valC.wzyx; // Reverse rendering order direction in the right quadrants.
        }
        else {
            if(p.y>0.){ d12 = vec4(-ij.zw, ij.xy); val = 1; } // Top left.
            else { /*d12.xy = -ij.zw;*/ d12.zw = ij.zw; p = p.yx; d12 = d12.yxwz;
                dirX *= -1.; val = 0; // Bottom left (Entry).
            }
        }

        // Ordering the cells from start to finish -- There's probably a smarter way,
        // but this is what I came up with at the time. It works, so it has that
        // going for it. :)
        //
        // The new quadrant value, after splitting, rotating, flipping, etc, above.
        int valN = p.x<0.? p.y<0.? 0 : 1 : p.y<0.? 3 : 2;
        // Number of squares per side for this iteration.
        int sL = 1<<(iters - i - 1); // 1, 2, 4, 8, etc.
        // Position number multiplied by total number of squares for each iteration.
        rn += valC[valN]*sL*sL;

        // Subdivide and center.
        p = mod(p, sc) - sc/2.;
        sc /= 2.;
    }

    // Square dimension. I.e. Number of blocks on the side of the square.
    float sL = float(1<<(iters - 1));

    // The distance field value.
    float d = 1e5;

    // If a swap occurred, swap the rendering order of dir1 and dir2.
    //if(valC[val] != val) d12 = d12.zwxy;

    // If a swap has occurred, reverse direction.
    float dir = valC[val] != val? -1. : 1.;

    float crvLR = 4./3.14159265; //Curve length ratio.

    // The two direction vectors in this cell are perpendicular.
    // Therefore, calculate the arc distance function and coordinates.
    // Otherwise, the direction vectors are aligned, so calculate
    // the line portion.
    //
    // By the way, for those who don't know, curvy line coordinates are
    // similar to 2D Euclidean plane coordinates. However, the X value runs
    // along the curve and the Y value is perpendicular to the curve.
    //
    if(dot(d12.xy, d12.zw) == 0.){
        // Arc distance field and the conversion of 2D plane coordinates
        // to curve coordinates.

        // Using the perpendicular direction vectors to center the arc.
        p -= (d12.xy + d12.zw)*sc;

        // Pixel angle.
        float a = atan(p.x, p.y);

        p.y = dist(p) - sc; // The Y coordinate (centered arc distance).

        d = abs(p.y); // Distance field value.

        p.x = fract(dir*a/6.2831853*4.); // The X coordinate (angle). Order counts.

        // Hacky distortion factor at the border of the line and arcs.
        //crvLR = mix(1.,  crvLR, 1. - p.x);
    }
    else {
        // Line distance field and curve coordinates.

        d = distLine(p, d12.xy*sc, d12.zw*sc); // Line distance (overshooting a bit).
        p.x = fract(dir*p.x*sL - .5); // Straight line coordinate.
        // p.y remains the same as the Euclidean Y value.

        // Hacky distortion factor at the border of the line and arcs.
        crvLR = mix(1., crvLR, smoothstep(0., 1., abs(p.x - .5)*2.));
        //crvLR = 1.;
    }

    // Using the current ordered cell value, the total number of cells and
    // the fractional curve cell value to calculate the overall ordered position
    // of the current pixel along the curve.
    float hPos = (float(rn) + p.x)/(sL*sL);

    // Getting rid of curves, etc, outside the rectangle domain.
    if(abs(op.x)>1. || abs(op.y)>.5){ d = 1e5; p = vec2(1e5); }

    // Handling (hacking) the entry and exit channels separately.
    if(op.y>.5){
        d = min(d, distLine(op - vec2(.5/sL, 0), vec2(0), ij.zw*4.)); hPos = 1. + (op.y - .5)/(sL);
        p.x = fract(op.y*sL); // Angle for this channel.
        crvLR = mix(1., crvLR, smoothstep(0., 1., abs(p.x - .5)*2.));
        p.y = (op.x - sc)*dirX;

    }
    if(op.y<-.5){
        d = min(d, distLine(op - vec2(-1. + .5/sL, 0), vec2(0), -ij.zw*4.)); hPos = (op.y + .5)/(sL);
        p.x = fract(op.y*sL); // Angle for this channel.
        crvLR = mix(1., crvLR, smoothstep(0., 1., abs(p.x - .5)*2.));
        p.y = -(op.x + (1. - sc))*dirX;

    }

    // The curve coordinates -- Scaled back to the zero to one range.
    p = vec2((p.x - .5)/sL/crvLR, p.y*dirX);

    // Line thickness.
    d -= .2/pow(1.6, float(iters));

    // Accounting for the left and right Hilbert curve blocks.
    if(op.x<0.){ hPos = 1. - fract(-hPos); p.y *= -1.; }

    // Return the distance field, curve position, and curve coordinates.
    return vec4(d/hSc, hPos, p/hSc);
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    // Aspect corret coordinates.
    vec2 uv = (fragCoord - iResolution.xy*.5)/iResolution.y;

    // Scale and smoothing factor.
    const float sc = 1.;
    float sf = sc/iResolution.y;

    // Scaling and translation.
    vec2 p = sc*uv;// + vec2(2, 1)*iTime/16.;

    // Scene field calculations.

    // Light direction. Shining down and to the left.
    vec2 ld = normalize(vec2(-1.5, -1));


    // Object shadow.
    vec4 dSh = hilbert(p - ld*.04);

    // Hightlight pass.
    vec4 d2 = hilbert(p - ld*.003);

    // Scene object.
    vec4 d = hilbert(p);

    // Using the two samples to produce some directional derivative based highlighting.
    float b = max(d2.x - d.x, 0.)/.003;
    float b2 = max(d.x - d2.x, 0.)/.003; // Reverse, for fake reflective light.
    //b = pow(b, 2.);

    // Background.
    vec2 ldUV = rot2(atan(-ld.x, ld.y))*uv;
    vec3 bg = mix(vec3(.92, .97, 1), vec3(.55, .7, 1), smoothstep(0., 1., ldUV.y*.35 + .5));

    // Scene color -- Set to the background.
    vec3 col = bg;

    // Adding subtle lines to the background.
    const float lnSc = 60.;
    vec2 pUV = rot2(-3.14159/4.)*p;
    float pat = (abs(fract(pUV.x*lnSc) - .5)*2. - .5)/lnSc/2.;
    col = mix(col, vec3(0), (1. - smoothstep(0., sf, pat))*.05);

    //b2 = mix(b2*.25, b2*1.25, smoothstep(0., sf, pat));
    //b = mix(b*1.25, b*.25, smoothstep(0., sf, pat));

    // Number of trails per Hilbert pattern block -- so six altogether.
    const float N = 3.;
    float sL = float(1<<(iters - 1)); // Block side length.
    float tm = iTime*sL/2.; // Movement.
    float crvL = sL*sL; // Block curve unit length.

    // The coordinates from the curves frame of reference.
    // X runs the entire length of the curve block, then wraps.
    // Y is simply the perpendicular coordinates with zero in the middle of the curve.
    vec2 tuv = d.yw;

    // The curve color.
    vec3 oCol = bg;
    // Color one side of the curve.
    //vec3 oCol = tuv.y<0.? bg : pow(bg.zyx, vec3(3));

    // Applying the bump highlighting to the curve color.
    oCol = min(oCol*(.45 + b*b*.75 + vec3(.5, .7, 1)*b2*.2), 1.5);
    vec3 svCol = oCol; // Saving the original.


    // Repeating space N times along a moving curve.
    // The value, stF, is a stretch factor relating to polar coordinate conversion.
    float stF = 4.;
    float cellID = floor(fract(tuv.x + tm/crvL)/( 1./N));
    float offsX = (hash21(vec2(cellID + 1., 1.23)) - .5)/N/4.;
    tuv = (mod(tuv + tm/crvL + offsX, 1./N) - .5/N)*stF;

    // Constructing a moving tip with a trail behind it.
    float trailL = 1./4./N*stF;
    float trail = abs(tuv.x) - trailL;
    float trailEnd = abs(tuv.x + (trailL + trailL/16.)) - trailL/16.;
    // Trail fade -- You could make this solid, if you wanted.
    float trailF = clamp((abs(tuv.x + trailL))/(trailL*2.), 0., 1.);
    trailF = trailF*.95 + .05;
    //float trailF = .1;

    // Trail tip color.
    vec3 tCol = svCol.z*mix(vec3(1, .1, .4), vec3(1, .4, .1), -uv.y*.5 + .5);

    // Applying the trail layers.
    //
    oCol = mix(oCol, svCol*trailF, 1. - smoothstep(0., sf, trail)); // Trail layer.
    // Alternative colored trail -- Comment in the above line first.
    //vec3 trC = mix(svCol*tCol/6., svCol, trailF);
    //oCol = mix(oCol, trC, 1. - smoothstep(0., sf, trail)); // Trail layer.
    oCol = mix(oCol, vec3(0), 1. -  smoothstep(0., sf, trailEnd)); // Dark trail end layer.
    oCol = mix(oCol, tCol*2., 1. - smoothstep(0., sf, trailEnd + .04/sL)); // Trail end layer.

    // Rendering onto the background.
    //
    col = mix(col, vec3(0), (1. - smoothstep(0., sf*4., dSh.x))*.35); // Shadow.
    //
    col = mix(col, vec3(0), (1. - smoothstep(0., sf*8., d.x))*.35); // AO.
    col = mix(col, vec3(0), (1. - smoothstep(0., sf, d.x))*.95); // Edge, or strke.
    col = mix(col, oCol, 1. - smoothstep(0., sf, d.x + .005)); // Top layer.

    /*
    // Dashes moving in opposite directions. Interesting, but a little much.
    float sc2 = float(1<<(iters - 3))*3.;
    float lns = (abs(fract(d.y*sc2*32. - (d.w<0.? -1. : 1.)*tm/8.) - .5) - .25)/32.;
    col = mix(col, col*.35, 1. - smoothstep(0., sf, max(abs(abs(d.w) - .0055), lns) - .005/3.));
    */

    /*
    // Experiment with segmenting cells and applying random coloring.
    stF = float(1<<(iters - 3))*3.;
    float scl = 1./3.;// 1./6.
    tuv = d.yw;
    tuv.x += tm/crvL + 1./stF*1. + offsX - 4./crvL;
    cellID = floor(fract(tuv.x)/scl);
    tuv.x = (mod(tuv.x, scl) - scl/2.)*stF;
    //float cObj = length(tuv) - float(1<<(iters - 3))*.85/crvL;
    float cObj = max(abs(tuv.x) - 4./crvL*stF, abs(tuv.y) - 2./crvL); //length(tuv) - 2.5/crvL;
    float crvTime = -iTime*(d.y<0.? -1. : 1.);

    float rnd = hash21(vec2(cellID, 1));
    vec3 cCol = .5 + .45*cos(6.2831*rnd/2. + vec3(1, 0, 2)*1.5);
    cCol *= .5 + b*b*.75;
    //if(mod(cellID, 2.)>.5) cObj = 1e5;
    col = mix(col, vec3(0), 1. - smoothstep(0., sf, cObj)); // Edge, or strke.
    col = mix(col, cCol, 1. - smoothstep(0., sf, cObj + .005)); // Edge, or strke.
    */

    // Rough gamma correction and screen plot.
    fragColor = vec4(sqrt(max(col, 0.)), 1);
}
