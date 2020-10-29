from r_theta import *

a, b = 2, 5

n = (10, 0.1, 1)

ellipse_weight = weight_reg(return_ellipse_reg(a,b), return_ellipse_reg_d1(a,b), return_ellipse_reg_d2(a,b))

smooth_weight = weight_reg(return_smooth_reg(*n), return_smooth_reg_d1(*n), return_smooth_reg_d2(*n))

third_weight = weight_reg(third_reg, third_reg_d1, third_reg_d2)

t = np.linspace(0, 2*np.pi, 600)

plt.polar(t, smooth_curve(t,*n), label=r'$r(\theta), n = 10, \varepsilon = 0.1$')# color='orange')
plt.legend(prop={'size': 20})
#plt.show()
plt.savefig('fig2.png', bbox_inches='tight')