# Meta-Learning approach for automatic kernel selection for support vector machines

Author:

- [Michael M.Meskhi](michaelmm.com)

This project is based on this paper by [Shawkat Ali and Kate A. Smith-Miles]().
By collection meta-features about a task at hand (dataset), we generate rules based on collected statistical information and perform automatic kernel selection for our Support Vector Machine.

Meta-features:

**1- Rule for polynomial kernel:**

```Python
if md <= 218.276:
	kernel = 'poly'
```

- *Mahanlanobis distance (md)*

![img](http://www.sciweavers.org/tex2img.php?eq=md_%7Brs%7D%5E%7B2%7D%3D%28x_%7Br%7Dx_%7Bs%7D%29V%5E%7B-1%7D%28x_%7Br%7Dx_%7Bs%7D%29%5E%7B%5Cprime%7D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)

**2- Rule for rbf kernel:**

```Python
elif (R > 9 and p-norm_cdf > 7.2957) or (p-du_cdf <= 2.8185):
	kernel = 'rbf'
```

- *Range (R)*

<img src="http://www.sciweavers.org/tex2img.php?eq=R%20%3D%20max%28X%29-min%28X%29&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0" align="center" border="0" alt="R = max(X)-min(X)" width="181" height="18" />

- *Normal cdf (P-norm_cdf)*

	<img src="https://latex.codecogs.com/gif.latex?P_{norm\_cdf} = F(x|\mu, \sigma) = \frac{1}{\sigma\sqrt{2\pi}}\int^{x}_{-\infty}e^{-(x-\mu)^{2}/2\sigma^{2}}dt" /> 

- *Discrete uniform cdf*

	<img src="https://latex.codecogs.com/gif.latex?P_{du\_cdf} = F(x|N) = \frac{\lfloor(x)\rfloor}{N}I_{(0,1...,n)}(x)" />

**3- Rule for Laplace kernel:**

```Python
elif y-gama_pdf <= 17.1671:
	kernel = 'laplace'
```

- *Gamma pdf*

	<img src="https://latex.codecogs.com/gif.latex?y_{gamma\_pdf} = f)(x|a, b) = \frac{1}{b^{a}\Gamma(a)}x^{a-1}e^{x/b}" /> 
	
**4- Rule for spline kernel:**

```Python
elif (y-ray_pdf > 20.2875) or (M <= 90.7233):
	kernel = 'spline'
```

- *Median (M)*

	<img src="https://latex.codecogs.com/gif.latex?M(a) = \frac{a_{\lfloor|x|/2\rfloor}+a_{\lfloor|x|/2+0.5\rfloor}}{2}" /> 
	
- *Rayleigh pdf*

	<img src="https://latex.codecogs.com/gif.latex?y_{ray\_pdf} = f(x|v) = \frac{\Gamma((v+1)/2)}{\Gamma(v/2)}\frac{1}{\sqrt{v\pi}}\frac{1}{(1+(x^{2}/v))^{(v+1)/2}}" /> 	
**5- Default rule if any of above are FALSE:**

```Python
else:
	kernel = 'rbf'
```