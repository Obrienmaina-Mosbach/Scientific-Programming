#determining if vectors are parallel
#if a vector is a scalar multiple of another vector, then they are parallel
#v1 = [a,b,c] and v2 = [d,e,f]
#v1 = k*v2
#where k is a scalar
#k = a/d = b/e = c/f
#v1 and v2 are parallel if k is a constant
#v1 and v2 are not parallel if k is not a constant
#v1 and v2 are parallel if they are scalar multiples of each other
#v1 and v2 are parallel if they have the same direction

#v1 = [2,3,4] and v2 = [4,6,8]
#v1 = 2*v2
#v1 and v2 are parallel
#v1 = [2,3,4] and v2 = [4,6,9]
#v1 = 2*v2
#v1 and v2 are not parallel

v1 = [2,3,4]
v2 = [4,6,8]
k = v1[0]/v2[0] == v1[1]/v2[1] == v1[2]/v2[2]
print(k)