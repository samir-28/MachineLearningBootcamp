import numpy as np

x=np.array([2,4,6,8,10])
y=np.array([6,5,4,3,2])



def regression(x,b0=7,b1=-0.5):
   return b0 + b1*x

n=len(x)
sum_x=np.sum(x) 
sum_y=np.sum(y) 
sum_xy=np.sum(x*y)
sum_xsq=np.sum(x**2)

b1= (n*sum_xy - sum_x*sum_y)/(n*sum_xsq - (sum_x**2))
b0= np.mean(y)-b1*np.mean(x)
# print(sum_x)
# print(sum_y)
# print(sum_xy)
# print(sum_xsq)
# print(b1)
# print(b0)
# print(f'Y={b0}{b1}X')
a=int(input("Enter the value of X :"))
print(f'When X=5 ,Y = {regression(a)}')