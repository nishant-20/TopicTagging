#!/usr/bin/env python
# coding: utf-8

# In[43]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import cv2
import pytesseract
import os
from PIL import Image
import sys


# In[44]:


def func(text):
    document = ["Electric Charges and Fields Like Charges and Unlike Charges Like charges repel and unlike charges attract each other Conductors and Insulators Conductors allow movement of electric charge through them insulators do not Quantization of Electric Charge It means that total charge of a body is always an integral multiple of a basic quantum of charge where   Additivity of Electric Charges Total charge of a system is the algebraic sum of all individual charges in the system Conservation of Electric Charges The total charge of an isolated system remains uncharged with time Superposition Principle It is the properties of forces with which two charges attractor repel each other are not affected by the presence of a third additional charge The Electric Field at a point due to a charge configuration It is the force on asmall positive test charge placed at the point divided by a magnitude and is given by It is radially outwards from if is positive and radially inwards if is negative at a point varies inversely as the square of its distance Coulomb’s Law The mutual electrostatic force between two point charges  and is proportional to the product  and inversely proportional to the square of the 2/3 distance separating them Where is a unit vector in the direction from to and is the proportionality constant An Electric Field Line It is a curve drawn in such a way that the tangent at each point on the curve gives the direction of electric field at that point Important Properties of Field Lines Field lines are continuous curves without any breaks Two field lines cannot cross each other Electrostatic field lines start at positive charges and end at negative charges They cannot form closed loops Electric Field at a Point due to Charge Electric Field due to an electric dipole in its equatorial plane at a distance from the centre Electric Field due to an Electric Dipole on the Axis at a Distance from theCentre  for Dipole Placed in Uniform Electric Field experiences a torque but experience no net force The Electric Flux is a dot product hence it is scalar is positive for all values of is negative for all values of Gauss’s Law The flux of electric field through any closed surface is times the total charge enclosed by Electric field due to an infinitely long straight wire of uniform linear charge density where r is the perpendicular distance of the point from the wire and  is the radial unit vector in the plane normal to the wire passing through the point Electric field due to an infinite thin plane sheet of uniform surface charge density where is a unit vector normal to the plane outward on either side Electric field due to thin spherical shell of uniform surface charge density for where r is the distance of the point from the centre of the shell and the radius of the shell is the total charge of the shell Electric field outside the charged shell is as though the total charge is concentrated at the centre The same result is true for a solid sphere of uniform volume charge density The electric field is zero at all points inside a charged shell Electric field along the outward normal to the surface is zero and is the surface charge density Charges in a conductor can reside only at its surface  Potential is constant within and on the surface of a conductor In a cavity within a conductor the electric field is zero",
            "Reflection When light is incident on a surface it is sent back by the surface in the same medium through which it had come This phenomenon is called 'reflection of light' by the surface Laws of Reflection The reflection at a plane surface always takes place in accordance with the following two laws The incident ray the reflected ray and normal to surface at the point of incidence all lie in the same plane The angle of incidence is equal to the angle of reflection Formation of Image by the Plane Mirror The formation of image of a point object by a plane mirror is represented in figure The image formed has the following characteristics The size of image is equal to the size of object The object distance Image distance The image is virtual and erect When a mirror is rotated through a certain angle the reflected ray is rotatedt hrough twice this angle Reflection of Light from Spherical Mirror A spherical mirror is a part cut from a hollow sphere They are generally constructed from glass  The reflection at spherical mirror also takes place in accordance with the laws of reflection Sign Convention Following sign conventions are the new cartesian sign convention All distances are measured from the pole of the mirror & the distances measured inthe direction of the incident light is taken as positive  In other words the distances measured toward the right of the origin are positive The distance measured against the direction of the incident light are taken as negative In other words the distances measured towards the left of origin are taken as negative The distance measured in the upward direction perpendicular to the principal axis of the mirror are taken as positive the distances measured in the downward direction are taken as negative Focal Length of a Spherical Mirror The distance between the focus and the pole of the mirror is called focal length ofthe mirror and is represented by The focal length of a concave mirror is negative and that of a convex mirror is positive The focal length of a mirror is equal to half of the radius of curvature of the mirror where is the radius of curvature of the mirror Principal Axis of the Mirror The straight line joining the pole and the centre of curvature of spherical mirror extended on both sides is called principal axis of the mirror",
            "Properties of magnets Attractive property Directive property Magnetic materials tend to point in the north southdirection Like magnetic poles repel and unlike ones attract Magnetic poles cannot be isolated always exists in pairs Magnetic induction A magnet induces magnetism in a magnetic substance placed near it This phenomenon is called magnetic induction Magnetic field lines It is defined as the curve the tangent to which at any point gives the direction of the magnetic field at that point Magnetic dipole An arrangement of two equal and opposite magnetic poles separated by a small distance is called a magnetic dipole Magnetic dipole moment It is defined as the product of its pole strength and magnetic length The force on it is zero The torque on it is Torque on dipole will be minimum when Torque on dipole will be maximum when When a bar magnet of dipole moment is placed in a uniform magnetic field then its potential energy is where we choose the zero of energy at theorientation when  is perpendicular to  On the axial point of the dipole Consider a bar magnet of size and magnetic moment m at a distance r from its midpoint the magnetic field  at an axial point P is Current loop as a magnetic dipole where I current flowing in the loop and A area of the loop Gauss’s Law for Magnetism It states that the net magnet flux through any closed surface is zeroPoles The pole near the geographic north pole of the earth is called the magnetic southpole The pole near the geographic south pole is called the magnetic north pole The magnitude of the magnetic field on the earth’s surface T Elements of the Earth’s Magnetic Field Three quantities are needed to specify the magnetic field of the earth on its surface The horizontal component of earth's magnetic field It is the component of the earth's total magnetic field in the horizontal direction in the magneticmeridian The magnetic declination The angle between the geographic meridian and the magnetic meridian at a place is called the magnetic declination at that place The angle of dip or magnetic inclination The angle made by the earth's total magnetic field  with the horizontal direction in the magnetic meridian is called angle of dip at any place",
            "Galileo extrapolated simple observations on motion of bodies on inclined planes and arrived at the law of inertia Newtons first law of motion is the same law rephrased thus Everybody continues to be in its state of rest or of uniform motion in a straight line unless compelled by some external force to act otherwise In simple terms the First Law is If external force on a body is zero its acceleration is zero Momentum of a body is the product of its mass and velocity Newtonís second law of motion The rate of change of momentum of a body is proportional to the applied force and takesplace in the direction in which the force acts Thus where F is the net external force on the body and a its acceleration We set the constant of proportionality in SI units Then The SI unit of force is newton The second law is consistent with the First Law It is a vector equation It is applicable to a particle and also to a body or a system of particles provided F is the total external force on the system and a is the acceleration of the system as a whole at a point at a certain instant determines a at the same point at that instant That is the Second Law is a local law a at an instant does not depend on the history of motion Impulse is the product of force and time which equals change in momentum The notion of impulse is useful when a large force acts for a short time to produce a measurable change in momentum Since the time of action of the force is very short one can assume that there is no appreciable change in the position of the body during the action of the impulsive force Newtons third law of motion To every action there is always an equal and opposite reaction In simple terms the law can be stated thus Forces in nature always occur between pairs of bodies Force on a body A by body B is equaland opposite to the force on the body B by Action and reaction forces are simultaneous forces There is no cause-effect relation between action and reaction Any of the two mutual forces can be called action and the other reaction Action and reaction act on different bodies and so they cannot be cancelled out The internal action and reaction forces between different parts of a body do however sum to zero Frictional force opposes relative motion between two surfaces incontact It is the component of the contact force along the common tangent to the surface incontact Static friction opposes impending relative motion kinetic friction opposes actual relative motion They are independentof the area of contact and satisfy the following",
            "Gravitation is the name given to the force of attraction acting between any two bodies of the universe law of gravitation It states that gravitational force of attraction acting between two point mass bodies of the universe is directly proportional to the product of their masses and is inversely proportional to the square of the distance between them where G is the universal gravitational constant Gravitational constant It is equal to the force of attraction acting between twobodies each of unit mass whose centres are placed unit distance apart Value of G is constant throughout the universe It is a scalar quantity The dimensional formula In SI unit the value of G Gravity It is the force of attraction exerted by earth towards its centre on a body lying on or near the surface of earth Gravity is the measure of weight of the body The weight of a body of mass acceleration due to gravity The unit of weight of a body will be the same as those of force Acceleration due to gravity It is defined as the acceleration set up in a bodywhile falling freely under the effect of gravity alone It is vector quantity The value of g changes with height depth rotation of earth the value of g is zero at the centre ofthe earth The value of g on the surface of earth is  The acceleration due togravity is related with gravitational constant by the relaion  where M and R are the mass and radius of the earth Gravitational potential: The gravitational potential at a point in a gravitational field is defined as the amount of work done in bringing a body of unit mass from infinity to that point without acceleration Gravitational potential at a point Gravitational intensity is related to gravitational potential at a point by the relationGravitational potential energy of a body at a point in the gravitational field of another body is defined as the amount of work done in bringing the given body from infinity to that point without acceleration",
            "Heat is a form of energy called thermal energy which flows from a higher temperature body to a lower temperature body when they are placed in contact Heat or thermal energy of a body is the sum of kinetic energies of all its constituent particles on account of translational vibrational and rotational motion The SI unit of heat energy is joule The practical unit of heat energy is calorie calorie is the quantity of heat required to raise the temperature of of water by Mechanical energy or work can be converted into heat Joule’s mechanical equivalent of heat J is a conversion factor and its value is Temperature Temperature of a body is the degree of hotness or coldness of the body A device which isused to measure the temperature is called a thermometer Highest possible temperature achieved in laboratory is about 108 while lowest possible temperature attained is Branch of Physics dealing with production and measurement temperature close to isknown as cryagenics while that deaf with the measurement of very high temperature is called pyromet temperature of the core of the sun is while that of its surface Different Scale of Temperature Celsius Scale In this scale of temperature the melting point ice is taken as and the boiling point of water as and space between these two points is divided into equal parts Fahrenheit Scale In this scale of temperature the melt point of ice is taken as and the boiling point of water as and the space between these two points is divided into equal parts Kelvin Scale In this scale of temperature the melting pouxl ice is taken as 273 K and the boiling point of water as 373 K the space between these two points is divided into 100 equal Thermometric Property The property of an object which changes with temperature is call thermometric property Different thermometric properties thermometers have been given below temperature coefficient of resistance and R0 and Rt are electrical resistances at 0C and tC resistance thermometer can measure temperature from"]
    
    document_topic = ["Electric Charges and Fields",
                 "Ray Optics",
                 "Magnetism",
                 "Laws of Motion",
                 "Gravitation",
                 "Thermal properties of Matter"]
    
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(document)
    
    true_k = len(document)
    model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=10)
    model.fit(X)
    
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    
    X = vectorizer.transform([text])
    predicted = model.predict(X)
    
    doc_topic_index=-1
    all_available = False
    
    for i in range(true_k):
        all_available=True
        
        for ind in order_centroids[predicted[0], :10]:
            if (terms[ind] not in document[i]):
                all_available=False
                break
        
        if (all_available):
            doc_topic_index=i
            break
    
    if (doc_topic_index == -1):
        return "null"
    else:
        return document_topic[doc_topic_index]


# In[45]:


def main(fileName):
    img = cv2.imread(fileName)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filename = "{}.png".format(os.getpid())
    cv2.imwrite(filename, gray)
    text = pytesseract.image_to_string(Image.open(filename))
    tag = func(text)
    print(tag)
    sys.stdout.write(tag)
    return tag
