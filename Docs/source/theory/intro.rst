.. _theory:

Introduction
============

.. figure:: Plasma_acceleration_sim.png
   :alt: Plasma laser-driven (top) and charged-particles-driven (bottom) acceleration (rendering from 3-D Particle-In-Cell simulations). A laser beam (red and blue disks in top picture) or a charged particle beam (red dots in bottom picture) propagating (from left to right) through an under-dense plasma (not represented) displaces electrons, creating a plasma wakefield that supports very high electric fields (pale blue and yellow). These electric fields, which can be orders of magnitude larger than with conventional techniques, can be used to accelerate a short charged particle beam (white) to high-energy over a very short distance.

   Plasma laser-driven (top) and charged-particles-driven (bottom) acceleration (rendering from 3-D Particle-In-Cell simulations). A laser beam (red and blue disks in top picture) or a charged particle beam (red dots in bottom picture) propagating (from left to right) through an under-dense plasma (not represented) displaces electrons, creating a plasma wakefield that supports very high electric fields (pale blue and yellow). These electric fields, which can be orders of magnitude larger than with conventional techniques, can be used to accelerate a short charged particle beam (white) to high-energy over a very short distance.

Computer simulations have had a profound impact on the design and understanding of past and present plasma acceleration experiments :cite:p:`in-Tsungpop06,in-Geddesjp08,in-Geddesscidac09,in-Geddespac09,in-Huangscidac09`, with
accurate modeling of wake formation, electron self-trapping and acceleration requiring fully kinetic methods (usually Particle-In-Cell) using large computational resources due to the wide range of space and time scales involved. Numerical modeling complements and guides the design and analysis of advanced accelerators, and can reduce development costs significantly. Despite the major recent experimental successes :cite:p:`in-LeemansPRL2014,in-Blumenfeld2007,in-BulanovSV2014,in-Steinke2016`, the various advanced acceleration concepts need significant progress to fulfill their potential. To this end, large-scale simulations will continue to be a key component toward reaching a detailed understanding of the complex interrelated physics phenomena at play.

For such simulations,
the most popular algorithm is the Particle-In-Cell (or PIC) technique,
which represents electromagnetic fields on a grid and particles by
a sample of macroparticles.
However, these simulations are extremely computationally intensive, due to the need to resolve the evolution of a driver (laser or particle beam) and an accelerated beam into a structure that is orders of magnitude longer and wider than the accelerated beam.
Various techniques or reduced models have been developed to allow multidimensional simulations at manageable computational costs: quasistatic approximation :cite:p:`in-Sprangle1990,in-Antonsenprl1992,in-Krallpre1993,in-Morapop1997,in-Quickpic`,
ponderomotive guiding center (PGC) models :cite:p:`in-Antonsenprl1992,in-Krallpre1993,in-Quickpic,in-Benedettiaac2010,in-Cowanjcp11`, simulation in an optimal Lorentz boosted frame :cite:p:`in-Vayprl07,in-Bruhwileraac08,in-Vayscidac09,in-Vaypac09,in-VayAAC2010,in-Martinscpc10,in-Martinsnaturephysics10,in-Martinspop10,in-Vayjcp2011,in-VayPOPL2011,in-Vaypop2011,in-Yu2016`,
expanding the fields into a truncated series of azimuthal modes :cite:p:`in-godfrey1985iprop,in-LifschitzJCP2009,in-DavidsonJCP2015,in-Lehe2016,in-AndriyashPoP2016`, fluid approximation :cite:p:`in-Krallpre1993,in-Shadwickpop09,in-Benedettiaac2010` and scaled parameters :cite:p:`in-Cormieraac08`.

.. bibliography::
   :keyprefix: in-
