//////////////////////////
// background.hpp
//////////////////////////
// 
// code components related to background evolution
//
// Author: Julian Adamek (Université de Genève & Observatoire de Paris & Queen Mary University of London)
//
// Last modified: September 2018
//
//////////////////////////

#ifndef BACKGROUND_HEADER
#define BACKGROUND_HEADER

#include <gsl/gsl_integration.h>
#include "metadata.hpp"


double FermiDiracIntegrand(double q, void * w)
{
	return q * q * sqrt(q * q + *(double *)w) / (exp(q) + 1.0l);
}

//////////////////////////
// FermiDiracIntegral
//////////////////////////
// Description:
//   computes the integral of the relativistic Fermi-Dirac distribution
// 
// Arguments:
//   w          parameter in the F-D distribution, "(m a / kB T)^2"
//
// Returns: value for the integral
// 
//////////////////////////

double FermiDiracIntegral(double &w)
{
	double result;
	gsl_function f;
	double err;
	size_t n;
	
	f.function = &FermiDiracIntegrand;
	f.params = &w;
		
	gsl_integration_qng(&f, 0.0l, 24.0l, 5.0e-7, 1.0e-7, &result, &err, &n);
	
	return result;
}


//////////////////////////
// bg_ncdm (1)
//////////////////////////
// Description:
//   computes the background model for one ncdm species by integrating the relativistic
//   Fermi-Dirac distribution
// 
// Arguments:
//   a          scale factor at which to compute the background model
//   cosmo      structure containing the cosmological parameters
//   p          index of the ncdm species
//
// Returns: value for the background model
// 
//////////////////////////

double bg_ncdm(const double a, const cosmology cosmo, const int p)
{
	if (p < 0 || p >= cosmo.num_ncdm)
		return 0;
	else
	{
		double w = a * cosmo.m_ncdm[p] / (pow(cosmo.Omega_g * cosmo.h * cosmo.h / C_PLANCK_LAW, 0.25) * cosmo.T_ncdm[p] * C_BOLTZMANN_CST);
		w *= w;
		
		return FermiDiracIntegral(w) * cosmo.Omega_ncdm[p] * pow(cosmo.Omega_g * cosmo.h * cosmo.h / C_PLANCK_LAW, 0.25) * cosmo.T_ncdm[p] * C_BOLTZMANN_CST / cosmo.m_ncdm[p] / C_FD_NORM / a;
	}
}


//////////////////////////
// bg_ncdm (2)
//////////////////////////
// Description:
//   computes the background model for all ncdm species by integrating the relativistic
//   Fermi-Dirac distribution
// 
// Arguments:
//   a          scale factor at which to compute the background model
//   cosmo      structure containing the cosmological parameters
//
// Note:
//   For optimization, the last value of a is stored in a static variable such that
//   multiple calls at the same value of a will not result in multiple integrations
//   being carried out. This assumes that the cosmological model should not change!
//
// Returns: value for the background model
// 
//////////////////////////

double bg_ncdm(const double a, const cosmology cosmo)
{
	double w;
	static double result = -1.0;
	static double a_prev = -1.0;
	
	if (a != a_prev)
	{
		result = 0.0;
		a_prev = a;
		
		for (int p = 0; p < cosmo.num_ncdm; p++)
			result += bg_ncdm(a, cosmo, p);
	}
	
	return result;
}


//////////////////////////
// Hconf
//////////////////////////
// Description:
//   computes the conformal Hubble rate at given scale factor
// 
// Arguments:
//   a          scale factor
//   fourpiG    "4 pi G"
//   cosmo      structure containing the cosmological parameters
//
// Returns: conformal Hubble rate
// 
//////////////////////////


double Omega_m(double &a, double &dm, double &rad, double &dcdm, const cosmology cosmo) { return ((dm+dcdm) / (dm+dcdm + cosmo.Omega_Lambda * a * a * a + rad / a )) ; }

double Omega_rad(double &a,  double &dm, double &rad, double &dcdm, const cosmology cosmo) { return ((rad) / (dm + dcdm * a + cosmo.Omega_Lambda * a * a * a * a + rad )) ; }

double Hconf(double a, double dm, double rad, double dcdm, const double fourpiG, const cosmology cosmo) 
{
    return sqrt((2. * fourpiG / 3.) * (Omega_m(a, dm, rad, dcdm, cosmo) + Omega_rad(a, dm, rad, dcdm, cosmo)+ cosmo.Omega_Lambda)* (a * a));
}

double d_a(double a, double dm, double rad, double dcdm, const double fourpiG, const cosmology cosmo) 
{
    return a * Hconf(a, dm, rad, dcdm, fourpiG, cosmo);
}

double d_dm(double a, double dm, double rad, double dcdm, const double fourpiG, const cosmology cosmo) 
{
    return -3 * Hconf(a, dm, rad, dcdm, fourpiG, cosmo) * dm;
}

double d_rad(double a, double dm, double rad, double dcdm, const double fourpiG, const cosmology cosmo) 
{
    return -4 * Hconf(a, dm, rad, dcdm, fourpiG, cosmo) * rad + a * cosmo.rate * dcdm;
}

double d_dcdm(double a, double dm, double rad, double dcdm, const double fourpiG, const cosmology cosmo) 
{
    return (-3 * Hconf(a, dm, rad, dcdm, fourpiG, cosmo) - a * cosmo.rate) * dcdm;
}

 


//////////////////////////
// rungekutta4bg
//////////////////////////
// Description:
//   integrates the Friedmann equation for the background model using a fourth-order
//   Runge-Kutta method
// 
// Arguments:
//   a          scale factor (will be advanced by dtau)
//   fourpiG    "4 pi G"
//   cosmo      structure containing the cosmological parameters
//   dtau       time step by which the scale factor should be advanced
//
// Returns:
// 
//////////////////////////

void rungekutta4bg(double &a, double &dm, double &rad, double &dcdm, const double fourpiG, const cosmology cosmo, double dtau) 

{
    vector<double> k1(4), k2(4), k3(4), k4(4);
    // k1
    k1[0] = d_a(a, dm, rad, dcdm, fourpiG, cosmo) * dtau;
    k1[1] = d_dm(a, dm, rad, dcdm, fourpiG, cosmo) * dtau;
    k1[2] = d_rad(a, dm, rad, dcdm, fourpiG, cosmo) * dtau;
    k1[3] = d_dcdm(a, dm, rad, dcdm, fourpiG, cosmo) * dtau;

    // k2
    k2[0] = d_a(a + k1[0] / 2, dm + k1[1] / 2, rad + k1[2] / 2,  dcdm + k1[3] / 2, fourpiG, cosmo) * dtau;
    k2[1] = d_dm(a + k1[0] / 2, dm + k1[1] / 2, rad + k1[2] / 2,  dcdm + k1[3] / 2, fourpiG, cosmo) * dtau;
    k2[2] = d_rad(a + k1[0] / 2, dm + k1[1] / 2, rad + k1[2] / 2,  dcdm + k1[3] / 2, fourpiG, cosmo) * dtau;
    k2[3] = d_dcdm(a + k1[0] / 2, dm + k1[1] / 2, rad + k1[2] / 2,  dcdm + k1[3] / 2, fourpiG, cosmo) * dtau;

    // k3
    k3[0] = d_a(a + k2[0] / 2, dm + k2[1] / 2, rad + k2[2] / 2, dcdm + k2[3] / 2, fourpiG, cosmo) * dtau;
    k3[1] = d_dm(a + k2[0] / 2, dm + k2[1] / 2, rad + k2[2] / 2, dcdm + k2[3] / 2, fourpiG, cosmo) * dtau;
    k3[2] = d_rad(a + k2[0] / 2, dm + k2[1] / 2, rad + k2[2] / 2, dcdm + k2[3] / 2, fourpiG, cosmo) * dtau;
    k3[3] = d_dcdm(a + k2[0] / 2, dm + k2[1] / 2, rad + k2[2] / 2, dcdm + k2[3] / 2, fourpiG, cosmo) * dtau;

    // k4
    k4[0] = d_a(a + k3[0], dm + k3[1], rad + k3[2], dcdm + k3[3], fourpiG, cosmo) * dtau;
    k4[1] = d_dm(a + k3[0], dm + k3[1], rad + k3[2], dcdm + k3[3], fourpiG, cosmo) * dtau;
    k4[2] = d_rad(a + k3[0], dm + k3[1], rad + k3[2], dcdm + k3[3], fourpiG, cosmo) * dtau;
    k4[3] = d_dcdm(a + k3[0], dm + k3[1], rad + k3[2], dcdm + k3[3], fourpiG, cosmo) * dtau;

    // Update state
    a += (k1[0] + 2*k2[0] + 2*k3[0] + k4[0]) / 6;
    dm += (k1[1] + 2*k2[1] + 2*k3[1] + k4[1]) / 6;
    rad +=(k1[2] + 2*k2[2] + 2*k3[2] + k4[2]) / 6;
    dcdm += (k1[3] + 2*k2[3] + 2*k3[3] + k4[3]) / 6;
    
}
    
    
//we define an initial Hconf because otherwise I do not know how to solve it as dm and rad etc 
//are only known at the discrete time steps that I get via rk4 loop


double initialparticleHorizonIntegrand(double sqrta, void * params)
{
        cosmology* cosmo = static_cast<cosmology*>(params);
        double dm = (1 - cosmo->dcdm_fraction) * cosmo->Omega_cdm;
        double rad = cosmo->Omega_rad;
        double dcdm = cosmo->dcdm_fraction * cosmo->Omega_cdm;

        return 2. / (sqrta * Hconf(sqrta*sqrta, dm, rad, dcdm, 1., *(cosmology *)cosmo));
}


//////////////////////////
// particleHorizon
//////////////////////////
// Description:
//   computes the particle horizon (tau) at given scale factor
// 
// Arguments:
//   a          scale factor
//   fourpiG    "4 pi G"
//   cosmo      structure containing the cosmological parameters
//
// Returns: particle horizon (tau)
// 
//////////////////////////

double particleHorizon(const double a, const double fourpiG, cosmology & cosmo)
{
	double result;
	gsl_function f;
	double err;
	size_t n;
	
	f.function = &initialparticleHorizonIntegrand;
	f.params = &cosmo;
	
	gsl_integration_qng(&f, sqrt(a) * 1.0e-7, sqrt(a), 5.0e-7, 1.0e-7, &result, &err, &n);
	
	return result / sqrt(fourpiG);
}

#endif

