This is the README file associated to the repository github.com/courbot/otmf.

Author : Jean-Baptiste Courbot

Website & contact informations: http://www.jb-courbot.fr/

This code is a prototype. If something is unclear or if you spot a bug, feel free to contact me !

**************************************
LICENSE

The code provided is licensed under the CeCILL-B licence available in Licence_CeCILL-B.txt. The implications are the following :
"CeCILL-B follows the principle of the popular BSD license and its variants (Apache, X11 or W3C among others). In exchange for 
strong citation obligations (in all software incorporating a program covered by CeCILL-B and also through a Web site), the author
authorizes the reuse of its software without any other constraints." 

When using these codes, please cite :
Courbot, J. B., Monfrini, E., Mazet, V.  & Collet, C. (2018). Oriented Triplet Markov Fields. Pattern Recognition Letters, 103, 16-22.

The code is a prototypical implementation of Oriented Triplet Markov Field (OTMF) model introduced in this paper.


**************************************
SETUP

To install this package, please run 
        python setup.py install
in the main package folder.

There is no specific package requirement, most of the code uses numpy and scipy modules.

Once installed, all package components are available using
        from otmf import *
        
**************************************
RUNNING

Once the installation is made, a demontration jupyter notebook "demo.ipynb" is available.
This file should be self-explanatory with regards to the input parameters and the methods from the paper.

**************************************
DOCUMENTATION

The sphinx-generated documentation is available from ./doc/_build/html/index.html.


**************************************
UPDATES AND CONTRIBUTIONS

The proposed codes are, with respect to the paper, in their most advanced form.
If you spot any bug, feel free to contact me (website above).
Any contribution on model & algorithms implementations are of course welcome.


**************************************
That's all folks !

