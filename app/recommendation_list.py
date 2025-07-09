import pandas as pd

data = [
    {
        "Topic": "Complex numbers",
        "Order": 1,
        "Concept": "Graphing on the complex plane",
        "Link": "https://www.khanacademy.org/math/algebra2/x2ec2f6f830c9fb89:complex/x2ec2f6f830c9fb89:complex-plane/v/plotting-complex-numbers-on-the-complex-plane",
    },
    {
        "Topic": "Complex numbers",
        "Order": 2,
        "Concept": "Finding roots of a quadratic equation",
        "Link": "https://www.khanacademy.org/math/algebra2/x2ec2f6f830c9fb89:complex/x2ec2f6f830c9fb89:complex-eq/v/complex-roots-from-the-quadratic-formula",
    },
    {
        "Topic": "Calculus",
        "Order": 1,
        "Concept": "Derivative of polynomial equations",
        "Link": "https://www.khanacademy.org/math/ap-calculus-ab/ab-differentiation-1-new/ab-2-6a/v/derivative-properties-and-polynomial-derivatives",
    },
    {
        "Topic": "Calculus",
        "Order": 2,
        "Concept": "Derivative of sine, cosine, exponential and logarithmic functions",
        "Link": "https://www.khanacademy.org/math/ap-calculus-ab/ab-differentiation-1-new/ab-2-7/v/derivatives-of-sinx-and-cosx",
    },
    {
        "Topic": "Calculus",
        "Order": 3,
        "Concept": "Standard limits",
        "Link": "https://www.khanacademy.org/math/ap-calculus-ab/ab-limits-new/ab-1-5b/v/limit-by-substitution",
    },
    {
        "Topic": "Calculus",
        "Order": 4,
        "Concept": "Limits at infinity",
        "Link": "https://www.khanacademy.org/math/ap-calculus-ab/ab-limits-new/ab-1-15/v/limits-at-positive-and-negative-infinity",
    },
    {
        "Topic": "Calculus",
        "Order": 5,
        "Concept": "Standard trigonometric limits",
        "Link": "https://www.khanacademy.org/math/ap-calculus-ab/ab-limits-new/ab-1-5b/v/limits-of-trigonometric-functions",
    },
    {
        "Topic": "Calculus",
        "Order": 6,
        "Concept": "Taylor approximations",
        "Link": "https://www.khanacademy.org/math/ap-calculus-bc/bc-series-new/bc-10-14/v/function-as-a-geometric-series",
    },
    {
        "Topic": "Calculus",
        "Order": 7,
        "Concept": "Indefinite integrals",
        "Link": "https://www.khanacademy.org/math/ap-calculus-ab/ab-integration-new/ab-6-7/v/antiderivatives-and-indefinite-integrals",
    },
    {
        "Topic": "Calculus",
        "Order": 8,
        "Concept": "Definite integrals",
        "Link": "https://www.khanacademy.org/math/ap-calculus-ab/ab-applications-of-integration-new/ab-8-4/v/evaluating-simple-definite-integral",
    },
    {
        "Topic": "Linear algebra",
        "Order": 1,
        "Concept": "Linear Equations",
        "Link": "https://www.khanacademy.org/math/test-prep/v2-sat-math/x0fcc98a58ba3bea7:algebra-harder/x0fcc98a58ba3bea7:solving-linear-equations-and-inequalities-harder/a/v2-sat-lesson-solving-linear-equations-and-inequalities",
    },
    {
        "Topic": "Linear algebra",
        "Order": 2,
        "Concept": "Matrix multiplication",
        "Link": "https://www.khanacademy.org/math/algebra-home/alg-matrices/alg-multiplying-matrices-by-matrices/v/matrix-multiplication-intro",
    },
    {
        "Topic": "Linear algebra",
        "Order": 3,
        "Concept": "2x2 Inverses",
        "Link": "https://www.khanacademy.org/math/algebra-home/alg-matrices/alg-finding-inverse-matrix-with-determinant/v/inverse-of-a-2x2-matrix",
    },
    {
        "Topic": "Linear algebra",
        "Order": 4,
        "Concept": "Matrix vector products",
        "Link": "https://www.khanacademy.org/math/linear-algebra/vectors-and-spaces/null-column-space/v/matrix-vector-products",
    },
    {
        "Topic": "Linear algebra",
        "Order": 5,
        "Concept": "Dot products",
        "Link": "https://www.khanacademy.org/math/linear-algebra/vectors-and-spaces/dot-cross-products/v/vector-dot-product-and-vector-length",
    },
    {
        "Topic": "Trigonometry",
        "Order": 1,
        "Concept": "Graphs of sin(x) and cos(x)",
        "Link": "https://www.khanacademy.org/math/algebra2/x2ec2f6f830c9fb89:trig/x2ec2f6f830c9fb89:trig-graphs/v/we-graphs-of-sine-and-cosine-functions",
    },
    {
        "Topic": "Trigonometry",
        "Order": 2,
        "Concept": "Evaluating functions at special angles",
        "Link": "https://www.khanacademy.org/math/algebra2/x2ec2f6f830c9fb89:trig/x2ec2f6f830c9fb89:special-angles/v/solving-triangle-unit-circle",
    },
    {
        "Topic": "Trigonometry",
        "Order": 3,
        "Concept": "Angle addition identities",
        "Link": "https://www.khanacademy.org/math/precalculus/x9e81a4f98389efdf:trig/x9e81a4f98389efdf:angle-addition/v/trigonometry-identity-review-fun",
    },
    {
        "Topic": "Real functions",
        "Order": 1,
        "Concept": "Intercepts",
        "Link": "https://www.khanacademy.org/math/algebra2/x2ec2f6f830c9fb89:poly-graphs",
    },
    {
        "Topic": "Real functions",
        "Order": 2,
        "Concept": "Exponential properties",
        "Link": "https://www.khanacademy.org/kmap/numerical-algebraic-expressions-c/xac37ebf62bd9d0f9:exponent-properties",
    },
    {
        "Topic": "Real functions",
        "Order": 3,
        "Concept": "Logarithm properties",
        "Link": "https://www.khanacademy.org/math/algebra2/x2ec2f6f830c9fb89:logs/x2ec2f6f830c9fb89:log-prop/v/introduction-to-logarithm-properties",
    },
    {
        "Topic": "Real functions",
        "Order": 4,
        "Concept": "Graphing logarithm and exponential",
        "Link": "https://www.khanacademy.org/math/algebra2/x2ec2f6f830c9fb89:logs/x2ec2f6f830c9fb89:log-intro/v/logarithms",
    },
]

RECOMMENDATION_LIST = pd.DataFrame(data)