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

	<a href="http://www.codecogs.com/eqnedit.php?latex=md_{rs}^{2}=(x_{r}-x_{s})V^{-1}(x_{r}-x_{s})^{\prime}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?md_{rs}^{2}=(x_{r}-x_{s})V^{-1}(x_{r}-x_{s})^{\prime}" title="md_{rs}^{2}=(x_{r}-x_{s})V^{-1}(x_{r}-x_{s})^{\prime}" /></a>

**2- Rule for rbf kernel:**

```Python
elif (R > 9 and p-norm_cdf > 7.2957) or (p-du_cdf <= 2.8185):
    kernel = 'rbf'
```

- *Range (R)*

	<a href="http://www.codecogs.com/eqnedit.php?latex=R&space;=&space;max(X)-min(X)" target="_blank"><img src="http://latex.codecogs.com/gif.latex?R&space;=&space;max(X)-min(X)" title="R = max(X)-min(X)" /></a>

- *Normal cdf (P-norm_cdf)*

	<a href="http://www.codecogs.com/eqnedit.php?latex=P_{norm\_cdf}&space;=&space;F(x|\mu,&space;\sigma)&space;=&space;\frac{1}{\sigma\sqrt{2\pi}}\int^{x}_{-\infty}e^{-(x-\mu)^{2}/2\sigma^{2}}dt" target="_blank"><img src="http://latex.codecogs.com/gif.latex?P_{norm\_cdf}&space;=&space;F(x|\mu,&space;\sigma)&space;=&space;\frac{1}{\sigma\sqrt{2\pi}}\int^{x}_{-\infty}e^{-(x-\mu)^{2}/2\sigma^{2}}dt" title="P_{norm\_cdf} = F(x|\mu, \sigma) = \frac{1}{\sigma\sqrt{2\pi}}\int^{x}_{-\infty}e^{-(x-\mu)^{2}/2\sigma^{2}}dt" /></a> 

- *Discrete uniform cdf*

	<a href="http://www.codecogs.com/eqnedit.php?latex=P_{du\_cdf}&space;=&space;F(x|N)&space;=&space;\frac{\lfloor(x)\rfloor}{N}I_{(0,1...,n)}(x)" target="_blank"><img src="http://latex.codecogs.com/gif.latex?P_{du\_cdf}&space;=&space;F(x|N)&space;=&space;\frac{\lfloor(x)\rfloor}{N}I_{(0,1...,n)}(x)" title="P_{du\_cdf} = F(x|N) = \frac{\lfloor(x)\rfloor}{N}I_{(0,1...,n)}(x)" /></a>

**3- Rule for Laplace kernel:**

```Python
elif y-gama_pdf <= 17.1671:
    kernel = 'laplace'
```

- *Gamma pdf*

	<a href="http://www.codecogs.com/eqnedit.php?latex=y_{gamma\_pdf}&space;=&space;f(x|a,&space;b)&space;=&space;\frac{1}{b^{a}\Gamma(a)}x^{a-1}e^{x/b}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?y_{gamma\_pdf}&space;=&space;f(x|a,&space;b)&space;=&space;\frac{1}{b^{a}\Gamma(a)}x^{a-1}e^{x/b}" title="y_{gamma\_pdf} = f(x|a, b) = \frac{1}{b^{a}\Gamma(a)}x^{a-1}e^{x/b}" /></a>
	
**4- Rule for spline kernel:**

```Python
elif (y-ray_pdf > 20.2875) or (M <= 90.7233):
    kernel = 'spline'
```

- *Median (M)*

	<a href="http://www.codecogs.com/eqnedit.php?latex=M(a)&space;=&space;\frac{a_{\lfloor|x|/2\rfloor}&plus;a_{\lfloor|x|/2&plus;0.5\rfloor}}{2}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?M(a)&space;=&space;\frac{a_{\lfloor|x|/2\rfloor}&plus;a_{\lfloor|x|/2&plus;0.5\rfloor}}{2}" title="M(a) = \frac{a_{\lfloor|x|/2\rfloor}+a_{\lfloor|x|/2+0.5\rfloor}}{2}" /></a> 
	
- *Rayleigh pdf*

	<a href="http://www.codecogs.com/eqnedit.php?latex=y_{ray\_pdf}&space;=&space;f(x|v)&space;=&space;\frac{\Gamma((v&plus;1)/2)}{\Gamma(v/2)}\frac{1}{\sqrt{v\pi}}\frac{1}{(1&plus;(x^{2}/v))^{(v&plus;1)/2}}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?y_{ray\_pdf}&space;=&space;f(x|v)&space;=&space;\frac{\Gamma((v&plus;1)/2)}{\Gamma(v/2)}\frac{1}{\sqrt{v\pi}}\frac{1}{(1&plus;(x^{2}/v))^{(v&plus;1)/2}}" title="y_{ray\_pdf} = f(x|v) = \frac{\Gamma((v+1)/2)}{\Gamma(v/2)}\frac{1}{\sqrt{v\pi}}\frac{1}{(1+(x^{2}/v))^{(v+1)/2}}" /></a>	
	
**5- Default rule if any of above are FALSE:**

```Python
else:
    kernel = 'rbf'
```