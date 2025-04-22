#import "@preview/cetz:0.2.2":canvas,draw,plot
#set page(height: auto)

#page(height: 370pt)[

#align(center,[= 1. Digital Images, Edges, and Derivatives])
== 1.1 Define edges and provide examples to illustrate the concept
#block(
  inset: 8pt,
  radius: 4pt,
  stroke: gray,
  [
  $bold("Definition:")$ An edge represents a local discontinuity or rapid variation in pixel values within an image. This discontinuity often occurs where there is a boundary between different objects or regions in the scene, leading to a sudden change in intensity values across neighboring pixels.  
  
  ]
)

#canvas(

{import draw:*

rect((-2,-2),(3,3))
rect((-1, -1), (rel: (3, 3)), radius: .5, stroke: (thickness: 0.6pt,dash:"dashed"),fill: yellow) 
rect((-.5, -.5), (rel: (2, 2)),

  radius: (north-east: (100%, .5),
           south-west: (100%, .5), rest: .2),
   stroke: (thickness: 0.6pt,dash:"dashed"),fill: red)

 rect((0,0), (1,1),stroke: (thickness: 0.6pt,dash:"dashed"),fill: green)
})
This dashed borders are edges]
#pagebreak()

== 1.2  Explain how derivatives are utilized for edge detection in one and two dimensions.  Describe edge indicators and supplement your explanation with visual examples.
#block(
  inset: 8pt,
  radius: 4pt,
  stroke: gray,
  [
  In image processing, derivatives are utilized for edge detection by quantifying the rate of change of intensity or color in an image. In one dimension, the derivative of the image function with respect to the spatial coordinate provides information about how rapidly the intensity changes along a single axis. In two dimensions, partial derivatives with respect to both the horizontal and vertical directions capture changes in intensity or color across the entire image.

Edge detection algorithms analyze these derivatives to identify points or regions where the gradient magnitude exceeds a predefined threshold (
𝜏
τ). The gradient magnitude represents the strength of the change in intensity, and pixels with high gradient magnitudes, greater than or equal to the threshold (
$abs(gradient f(x)) >= tau $), are likely to be located at edges.
  ]
)

 Let: #text(fill:blue)[$f(t) = sin(t) + cos(sqrt(3) t)$]
and thereby
#text(fill: red)[$abs(f(t)') = abs(cos(t) - sqrt(3) sin(sqrt(3)t) )$]
with #text(fill: green)[$tau = 1$] we get edge on intervals
#text(fill: rgb(172, 229, 253))[$[-5.3,-4.2],[-3.6,-2],[-1.5,0],[0.8,1.34],[3.69,4.99]$]


#canvas({
  import draw:*
plot.plot(size:(10,10),axis-style:"left",



{
  plot.annotate({
  rect((-5.3,2.7),(-4.2,-1.99),stroke: none, fill: rgb(172, 229, 253))
   rect((-3.6,2.7),(-2,-1.99),stroke: none, fill: rgb(172, 229, 253))
   rect((-1.5,2.7),(0,-1.99),stroke: none, fill: rgb(172, 229, 253))
    rect((0.8,2.7),(1.34,-1.99),stroke: none, fill: rgb(172, 229, 253))
     rect((3.69,2.7),(4.99,-1.99),stroke: none, fill: rgb(172, 229, 253))},background:true)
  
  
  
  
  plot.add(domain: (-6.5,6.5), x => calc.sin(x) + calc.cos(calc.sqrt(3)*x))
plot.add(domain: (-6.5,6.5), x => calc.abs(calc.cos(x) - calc.sqrt(3)*calc.sin(calc.sqrt(3)*x)))
plot.add(domain:(-6,6), x => 1)
 })
})

#pagebreak()
Two dimensional example of Orange gutang
(Photo is a 2D function)
#figure(image("orangegutang.jpg"))
#figure(image("edgesorange_degree_1.jpg"))


#pagebreak()

== 1.3 Investigate the impact of truncation error in finite difference formulas on edge detection.  Support your findings with visual evidence.
#text(fill:blue)[$f(x) = sin(t) + cos(sqrt(3) t)$]
it's exact derivative is 
#text(fill: red)[$abs(f(x)') = abs(cos(t) - sqrt(3) sin(sqrt(3)t) )$]

and derivative with finite difference #text(fill:yellow, [$f(t)^' approx ((f(t+h)-f(t))/h)$]) degree one, we can clearly see that some edges from red function derivative are "missed" in yellow finite difference approximation due to truncation error


#canvas({
  import draw:*
plot.plot(size:(10,10),axis-style:"left",



{
  

  plot.add(domain: (-6.5,6.5), x => calc.sin(x) + calc.cos(calc.sqrt(3)*x))
plot.add(domain: (-6.5,6.5), x => calc.abs(calc.cos(x) - calc.sqrt(3)*calc.sin(calc.sqrt(3)*x)))
plot.add(domain:(-6,6), x => 1)

plot.add(domain: (-6.5,6.5), x => calc.abs((calc.sin(x+0.5) + calc.cos(calc.sqrt(3)*x + 0.5) - (calc.sin(x) + calc.cos(calc.sqrt(3)*x)))/0.5))

 })
})


#pagebreak()
= 2.Digital Images, Features, and Higher Order Derivatives

== 2.1-2  Investigate whether higher order derivatives can be employed for edge detection and Explore the potential of higher order derivatives in extracting other features of digital images.

Higher order derivatives can be utilized for image sharpening or blurring by following these steps:
#enum(enum.item(1)[$bold("Choose the appropriate order and degree:")$ Determine the derivative's order (e.g., second derivative for sharpening or blurring) and the degree of the central finite difference approximation (e.g., degree 2 for a second-order derivative)],
enum.item(2)[$bold("Define the finite difference kernel:")$ Construct a finite difference kernel that represents the desired derivative. For instance, for a second-order derivative in one dimension, a common kernel is [1, -2, 1], or its equivalent for higher degrees.],
enum.item(3)[$bold("Apply the finite difference kernel:")$ Utilize the central finite difference method to convolve the image with the finite difference kernel in the horizontal and vertical directions to compute the desired derivative.
],
enum.item(4)[ $bold("Compute the magnitude of the gradient:")$ Combine the derivatives calculated in both directions (horizontal and vertical) to determine the gradient's magnitude, representing the strength of edges in the image.],
enum.item(5)[ $bold("Thresholding and post-processing:")$ Optionally, apply thresholding or other post-processing techniques to refine the detected edges and reduce noise.

 By integrating higher order derivatives with central finite difference methods, it is possible to enhance the sensitivity and accuracy of edge detection algorithms, especially in scenarios where precise details or subtle changes in intensity are essential for identifying edges." ],)
  




 




  
#pagebreak()


== 2.3 Present visual examples to illustrate both successful and unsuccessful findings.
#grid(columns: (auto,200pt),
  rows: (auto, 150pt),
  gutter: 5pt,
  figure(image("edges_degree_1.jpg", width: 200pt), kind: "example",
  supplement: [Example],caption: "Degree 1"),
figure(image("edges_degree_2.jpg",width: 200pt), kind: "example",
  supplement: [Example],caption: "Degree 2"),
figure(image("edges_degree_3.jpg",width: 200pt), kind: "example",
  supplement: [Example],caption: "Degree 3"),
figure(image("edges_degree_4.jpg",width: 200pt), kind: "example",
  supplement: [Example],caption: "Degree 4"),
figure(image("edges_degree_5.jpg",width: 200pt), kind: "example",
  supplement: [Example],caption: "Degree 5"),
figure(image("edges_degree_6.jpg",width: 200pt), kind: "example",
  supplement: [Example],caption: "Degree 6"))

  #pagebreak()
  = 3 Exploring the Applications of Derivatives
  == 3.1  Provide two illustrations of how derivatives are applied in real-world scenarios.

  $bold(space space "Medical Imaging:")$ Edge detection is extensively used in medical imaging for various purposes, such as identifying anatomical structures, detecting tumors, and analyzing cellular structures. For instance, in X-ray or MRI images, edge detection algorithms can help identify boundaries between different tissues or organs, aiding radiologists in diagnosing diseases or abnormalities.



  $space space bold("Object Detection in Robotics: ")$ In robotics and computer vision applications, edge detection is commonly used for object detection and recognition. Robots equipped with cameras can use edge detection algorithms to identify objects in their environment based on their shapes and contours. This capability enables robots to navigate through complex environments, manipulate objects, and perform tasks autonomously, such as picking and placing items on assembly lines or in warehouses.
  #pagebreak()

#page(height: auto)[== 3.2 Ensure one of the examples is connected with the concept of edge detection.

$bold("Object Detection and Recognition:")$ With the help of edge-based features, robots can classify and recognize objects in their environment. By comparing extracted features with pre-defined object models or using machine learning algorithms, robots can determine the identity of detected objects. This information is essential for robotic tasks such as navigation, manipulation, and interaction with the environment.]


#pagebreak()
== 3.3 Present a visual representation for at least one of the examples.  Clarify why this application works and mention the tools that can be utilized for this purpose.

$bold("EXAMPLE WILL BE DURING PRESENTATION ^^")  $

Intro: #figure(image("kaiada.png"))


#pagebreak()
= 4 Digital Images, Features, and Linear Combinations of Derivatives

== 4.1 Investigate whether linear combinations of derivatives can be employed for edge detection.  Describe the method used to select coefficients in the linear combination of derivatives and justify your approach: 

$space space $Linear combinations of derivatives can indeed be employed for edge detection in digital images. The method involves combining multiple derivative filters with appropriate coefficients to enhance the detection of edges, which represent significant changes in intensity or color in an image. The formula for such a linear combination is: $f(x) = c_0 dot f(x) + c_1 dot (d f(x))/(d x) + c_2 dot (d^2 f(x))/(d x^2)+ ... + c_n dot (d^n f(x))/(d x^n)$ in this formula:

f(x) represents the input image or the function of interest.

$c_0, c_1, c_2, ..., c_n$ are the coefficients assigned to each derivative term.

$(d f(x))/(d x), (d^2 f(x))/(d x^2),...,(d^n f(x))/(d x^n) "are the first,second ... n-th order derivatives with respect to the variable x" $
To select coefficients for the linear combination, one approach involves empirical tuning or optimization techniques such as gradient descent. The coefficients are adjusted iteratively to maximize the edge detection performance based on predefined evaluation metrics such as edge detection accuracy, precision, and recall. Alternatively, coefficients can be set based on known mathematical properties of derivative filters, such as the Sobel or Prewitt operators, which are commonly used for edge detection. These coefficients are chosen to emphasize the contribution of each derivative term while minimizing noise and artifacts in the resulting edge map.

== 4.2 Explore the potential of linear combinations of derivatives in extracting other features of digital images.  Describe the method used to select coefficients in the linear combination of derivatives and justify your approach

$space space space space space$ Linear combinations of derivatives also offer potential for extracting various features from digital images beyond edge detection. By adjusting the coefficients in the linear combination formula, it's possible to highlight specific features such as textures, corners, or blobs. The formula remains the same as in edge detection but with coefficients tailored to target different characteristics of interest.

$space space space space space$coefficients are chosen to emphasize the detection of features other than edges. For example, to highlight textures, coefficients may be selected to enhance high-frequency components in the image. Similarly, to detect corners or keypoints, coefficients may be adjusted to emphasize second-order derivatives in multiple directions.The selection of coefficients depends on the specific features being targeted and the desired characteristics of the extracted features. Coefficients can be manually tuned based on domain knowledge or data-driven approaches such as machine learning algorithms trained to optimize feature extraction performance. Additionally, domain-specific constraints and considerations, such as computational efficiency and robustness to noise, should guide the selection process to ensure the effectiveness of feature extraction using linear combinations of derivatives.

== 4.3 Present visual examples to illustrate both successful and unsuccessful findings.

No need