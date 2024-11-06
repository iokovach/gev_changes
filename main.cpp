//////////////////////////
// Copyright (c) 2015-2019 Julian Adamek
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//  
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//  
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESSED OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//////////////////////////

//////////////////////////
// main.cpp
//////////////////////////
// 
// main control sequence of Geneva N-body code with evolution of metric perturbations (gevolution)
//
// Author: Julian Adamek (Université de Genève & Observatoire de Paris & Queen Mary University of London & Universität Zürich)
//
// Last modified: August 2022
//
//////////////////////////

#include <stdlib.h>
#include <set>
#include <vector>

#include "LATfield2.hpp"
#include "metadata.hpp"
#include "class_tools.hpp"
#include "tools.hpp"
#include "background.hpp"
#include "Particles_gevolution.hpp"
#include "gevolution.hpp"
#include "ic_basic.hpp"
#include "ic_read.hpp"

#include "radiation.hpp"
#include "parser.hpp"
#include "output.hpp"
#include "hibernation.hpp"


using namespace std;
using namespace LATfield2;

int main(int argc, char **argv)
{
#ifdef BENCHMARK
	//benchmarking variables
	double ref_time, ref2_time, cycle_start_time;
	double initialization_time;
	double run_time;
	double cycle_time=0;
	double projection_time = 0;
	double snapshot_output_time = 0;
	double spectra_output_time = 0;
	double lightcone_output_time = 0;
	double gravity_solver_time = 0;
	double fft_time = 0;
	int fft_count = 0;   
	double update_q_time = 0;
	int update_q_count = 0;
	double moveParts_time = 0;
	int  moveParts_count =0;	
#endif  //BENCHMARK
	
	int n = 0, m = 0;
	int io_size = 0;
	int io_group_size = 0;
	
	int i, j, cycle = 0, snapcount = 0, pkcount = 0, restartcount = 0, usedparams, numparam = 0, numspecies, done_hij;
	int numsteps_ncdm[MAX_PCL_SPECIES-2];
	long numpts3d;
	int box[3];
	double dtau, dtau_old, dx, tau, a, dm, rad, dcdm, fourpiG, tmp, start_time;
	double maxvel[MAX_PCL_SPECIES];
	FILE * outfile;
	char filename[2*PARAM_MAX_LENGTH+24];
	string h5filename;
	char * settingsfile = NULL;
	char * precisionfile = NULL;
	parameter * params = NULL;
	metadata sim;
	cosmology cosmo;
	icsettings ic;
	double T00hom;


	
	for (i=1 ; i < argc ; i++ ){
		if ( argv[i][0] != '-' )
			continue;
		switch(argv[i][1]) {
			case 's':
				settingsfile = argv[++i]; //settings file name
				break;
			case 'n':
				n = atoi(argv[++i]); //size of the dim 1 of the processor grid
				break;
			case 'm':
				m =  atoi(argv[++i]); //size of the dim 2 of the processor grid
				break;
			case 'p':

				precisionfile = argv[++i];
				break;
			case 'i':
#ifndef EXTERNAL_IO
				cout << "EXTERNAL_IO needs to be set at compilation to use the I/O server"<<endl;
				exit(-1000);
#endif
				io_size =  atoi(argv[++i]);
				break;
			case 'g':
#ifndef EXTERNAL_IO
				cout << "EXTERNAL_IO needs to be set at compilation to use the I/O server"<<endl;
				exit(-1000);
#endif
				io_group_size = atoi(argv[++i]);
		}
	}

#ifndef EXTERNAL_IO
	parallel.initialize(n,m);
#else
	if (!io_size || !io_group_size)
	{
		cout << "invalid number of I/O tasks and group sizes for I/O server (-DEXTERNAL_IO)" << endl;
		exit(-1000);
	}
	parallel.initialize(n,m,io_size,io_group_size);
	if(parallel.isIO()) ioserver.start();
	else
	{
#endif
	
	COUT << COLORTEXT_WHITE << endl;	
	COUT << "  _   _      _         __ ,  _" << endl;
	COUT << " (_| (-' \\/ (_) (_ (_| (  ( (_) /\\/	version 1.2         running on " << n*m << " cores." << endl;
	COUT << "  -'" << endl << COLORTEXT_RESET << endl;
	
#if GRADIENT_ORDER > 1
	COUT << " compiled with GRADIENT_ORDER=" << GRADIENT_ORDER << endl;
#endif
	
	if (settingsfile == NULL)
	{
		COUT << COLORTEXT_RED << " error" << COLORTEXT_RESET << ": no settings file specified!" << endl;
		parallel.abortForce();
	}
	
	COUT << " initializing..." << endl;
	
	start_time = MPI_Wtime();
	
	numparam = loadParameterFile(settingsfile, params);
	
	usedparams = parseMetadata(params, numparam, sim, cosmo, ic);
	
	COUT << " parsing of settings file completed. " << numparam << " parameters found, " << usedparams << " were used." << endl;
	
	sprintf(filename, "%s%s_settings_used.ini", sim.output_path, sim.basename_generic);
	saveParameterFile(filename, params, numparam);
	
	free(params);


		numparam = 0;
	
	h5filename.reserve(2*PARAM_MAX_LENGTH);
	h5filename.assign(sim.output_path);
	
	box[0] = sim.numpts;
	box[1] = sim.numpts;
	box[2] = sim.numpts;
	
	Lattice lat(3,box,GRADIENT_ORDER);
	Lattice latFT;
	latFT.initializeRealFFT(lat,0);
	
	Particles_gevolution<part_simple,part_simple_info,part_simple_dataType> pcls_cdm;
	Particles_gevolution<part_simple,part_simple_info,part_simple_dataType> pcls_b;
	Particles_gevolution<part_simple,part_simple_info,part_simple_dataType> pcls_ncdm[MAX_PCL_SPECIES-2];
	Field<Real> * update_cdm_fields[3];
	Field<Real> * update_b_fields[3];
	Field<Real> * update_ncdm_fields[3];
	double f_params[5];
	set<long> IDbacklog[MAX_PCL_SPECIES];

	Field<Real> phi;
	Field<Real> source;
	Field<Real> chi;
	Field<Real> Sij;
	Field<Real> Bi;
	Field<Cplx> scalarFT;
	Field<Cplx> SijFT;
	Field<Cplx> BiFT;
	source.initialize(lat,1);
	phi.initialize(lat,1);
	chi.initialize(lat,1);
	scalarFT.initialize(latFT,1);
	PlanFFT<Cplx> plan_source(&source, &scalarFT);
	PlanFFT<Cplx> plan_phi(&phi, &scalarFT);
	PlanFFT<Cplx> plan_chi(&chi, &scalarFT);
	Sij.initialize(lat,3,3,symmetric);
	SijFT.initialize(latFT,3,3,symmetric);
	PlanFFT<Cplx> plan_Sij(&Sij, &SijFT);
	Bi.initialize(lat,3);
	BiFT.initialize(latFT,3);
	PlanFFT<Cplx> plan_Bi(&Bi, &BiFT);

	update_cdm_fields[0] = &phi;
	update_cdm_fields[1] = &chi;
	update_cdm_fields[2] = &Bi;
	
	update_b_fields[0] = &phi;
	update_b_fields[1] = &chi;
	update_b_fields[2] = &Bi;
	
	update_ncdm_fields[0] = &phi;
	update_ncdm_fields[1] = &chi;
	update_ncdm_fields[2] = &Bi;
	
	Site x(lat);
	rKSite kFT(latFT);
	
	dx = 1.0 / (double) sim.numpts;
	numpts3d = (long) sim.numpts * (long) sim.numpts * (long) sim.numpts;
	
	for (i = 0; i < 3; i++) // particles may never move farther than to the adjacent domain
	{
		if (lat.sizeLocal(i)-1 < sim.movelimit)
			sim.movelimit = lat.sizeLocal(i)-1;
	}
	parallel.min(sim.movelimit);

	fourpiG = 1.5 * sim.boxsize * sim.boxsize / C_SPEED_OF_LIGHT / C_SPEED_OF_LIGHT;
	a = 1. / (1. + sim.z_in);
	dm = (1 - cosmo.dcdm_fraction) * cosmo.Omega_cdm;
	rad = cosmo.Omega_rad;
	dcdm = cosmo.dcdm_fraction * cosmo.Omega_cdm;
	COUT << "initial sum of densities =" << dm + rad + dcdm + cosmo.Omega_Lambda << endl;
//tau = particleHorizon(a, fourpiG, cosmo);
	tau = ic.tau_in; 
	COUT << "initial volume times critical density = " <<  3* Hconf(a, dm, rad, dcdm, fourpiG, cosmo) * Hconf(a, dm, rad, dcdm, fourpiG, cosmo)/(2*fourpiG*a*a) << endl;
	
	if (sim.Cf * dx < sim.steplimit / Hconf(a, dm, rad, dcdm, fourpiG, cosmo))
		dtau = sim.Cf * dx;
	else
		dtau = sim.steplimit / Hconf(a, dm, rad, dcdm, fourpiG, cosmo);
		
	dtau_old = 0.;
	
	if (ic.generator == ICGEN_BASIC)
		generateIC_basic(sim, ic, cosmo, fourpiG, &pcls_cdm, &pcls_b, pcls_ncdm, maxvel, &phi, &chi, &Bi, &source, &Sij, &scalarFT, &BiFT, &SijFT, &plan_phi, &plan_chi, &plan_Bi, &plan_source, &plan_Sij, params, numparam); // generates ICs on the fly
	else if (ic.generator == ICGEN_READ_FROM_DISK)
		readIC(sim, ic, cosmo, fourpiG, a, dm, rad, dcdm, tau, dtau, dtau_old, &pcls_cdm, &pcls_b, pcls_ncdm, maxvel, &phi, &chi, &Bi, &source, &Sij, &scalarFT, &BiFT, &SijFT, &plan_phi, &plan_chi, &plan_Bi, &plan_source, &plan_Sij, cycle, snapcount, pkcount, restartcount, IDbacklog);

	else
	{
		COUT << " error: IC generator not implemented!" << endl;
		parallel.abortForce();
	}

	if (sim.baryon_flag > 1)
	{
		COUT << " error: baryon_flag > 1 after IC generation, something went wrong in IC generator!" << endl;
		parallel.abortForce();
	}
	
	numspecies = 1 + sim.baryon_flag + cosmo.num_ncdm;	
	parallel.max<double>(maxvel, numspecies);
	
	if (sim.gr_flag > 0)
	{
		for (i = 0; i < numspecies; i++)
			maxvel[i] /= sqrt(maxvel[i] * maxvel[i] + 1.0);
	}


	
#ifdef BENCHMARK
	initialization_time = MPI_Wtime() - start_time;
	parallel.sum(initialization_time);
	COUT << COLORTEXT_GREEN << " initialization complete." << COLORTEXT_RESET << " BENCHMARK: " << hourMinSec(initialization_time) << endl << endl;
#else
	COUT << COLORTEXT_GREEN << " initialization complete." << COLORTEXT_RESET << endl << endl;
#endif



	while (true)    // main loop
	{

		// construct stress-energy tensor
		projection_init(&source);

		if (sim.gr_flag > 0)
		{
			projection_T00_project(&pcls_cdm, &source, a, &phi);
			if (sim.baryon_flag)
				projection_T00_project(&pcls_b, &source, a, &phi);
			for (i = 0; i < cosmo.num_ncdm; i++)
			{
				if (a >= 1. / (sim.z_switch_deltancdm[i] + 1.) && sim.numpcl[1+sim.baryon_flag+i] > 0)
					projection_T00_project(pcls_ncdm+i, &source, a, &phi);
				else if (sim.radiation_flag == 0 || (a >= 1. / (sim.z_switch_deltancdm[i] + 1.) && sim.numpcl[1+sim.baryon_flag+i] == 0))
				{
					tmp = bg_ncdm(a, cosmo, i);
					for(x.first(); x.test(); x.next())
						source(x) += tmp;
				}
			}
		}
		else
		{
			scalarProjectionCIC_project(&pcls_cdm, &source);
			if (sim.baryon_flag)
				scalarProjectionCIC_project(&pcls_b, &source);
			for (i = 0; i < cosmo.num_ncdm; i++)
			{
				if (a >= 1. / (sim.z_switch_deltancdm[i] + 1.) && sim.numpcl[1+sim.baryon_flag+i] > 0)
					scalarProjectionCIC_project(pcls_ncdm+i, &source);
			}
		}
		projection_T00_comm(&source);
		


		if (sim.vector_flag == VECTOR_ELLIPTIC)
		{
			projection_init(&Bi);
			projection_T0i_project(&pcls_cdm, &Bi, &phi);
			if (sim.baryon_flag)
				projection_T0i_project(&pcls_b, &Bi, &phi);
			for (i = 0; i < cosmo.num_ncdm; i++)
			{
				if (a >= 1. / (sim.z_switch_Bncdm[i] + 1.) && sim.numpcl[1+sim.baryon_flag+i] > 0)
					projection_T0i_project(pcls_ncdm+i, &Bi, &phi);
			}
			projection_T0i_comm(&Bi);
		}
		
		projection_init(&Sij);
		projection_Tij_project(&pcls_cdm, &Sij, a, &phi);
		if (sim.baryon_flag)
			projection_Tij_project(&pcls_b, &Sij, a, &phi);
		if (a >= 1. / (sim.z_switch_linearchi + 1.))
		{
			for (i = 0; i < cosmo.num_ncdm; i++)
			{
				if (sim.numpcl[1+sim.baryon_flag+i] > 0)
					projection_Tij_project(pcls_ncdm+i, &Sij, a, &phi);
			}
		}
		projection_Tij_comm(&Sij);
		

		
		if (sim.gr_flag > 0)
		{	
			T00hom = 0.;
			for (x.first(); x.test(); x.next())
				T00hom += source(x);
			parallel.sum<double>(T00hom);
			T00hom /= (double) numpts3d;
			
			if (cycle % CYCLE_INFO_INTERVAL == 0)
			{
				COUT << " cycle " << cycle << ", background information: z = " << (1./a) - 1. << ", average T00 = " << T00hom << ", background model = " << dm+dcdm << endl;
			}
			
			if (dtau_old > 0.)
			{
				prepareFTsource<Real>(phi, chi, source, Omega_m(a, dm, rad, dcdm, cosmo), source, 3. * Hconf(a, dm, rad, dcdm, fourpiG, cosmo) * dx * dx / dtau_old, fourpiG * dx * dx / a, 3. * Hconf(a, dm, rad, dcdm, fourpiG, cosmo) * Hconf(a, dm, rad, dcdm, fourpiG, cosmo) * dx * dx);  // prepare nonlinear source for phi update


				plan_source.execute(FFT_FORWARD);  // go to k-space

		
				solveModifiedPoissonFT(scalarFT, scalarFT, 1. / (dx * dx), 3. * Hconf(a, dm, rad, dcdm, fourpiG, cosmo) / dtau_old);  // phi update (k-space)
	
				plan_phi.execute(FFT_BACKWARD);	 // go back to position space

	
			}
		}
		else
		{

			plan_source.execute(FFT_FORWARD);  // Newton: directly go to k-space

		
			solveModifiedPoissonFT(scalarFT, scalarFT, fourpiG / a);  // Newton: phi update (k-space)

	
			plan_phi.execute(FFT_BACKWARD);	 // go back to position space

		}

		phi.updateHalo();  // communicate halo values

		// record some background data
		if (kFT.setCoord(0, 0, 0))
		{
			sprintf(filename, "%s%s_background.dat", sim.output_path, sim.basename_generic);
			outfile = fopen(filename, "a");
			if (outfile == NULL)
			{
				cout << " error opening file for background output!" << endl;
			}
			else
			{
				if (cycle == 0)
					fprintf(outfile, "# background statistics\n# cycle   tau/boxsize    a     dm     rad     dcdm      conformal H/H0  phi(k=0)       T00(k=0)\n");
				fprintf(outfile, " %8d   %e   %e   %e   %e   %e   %e   %e\n", cycle, tau, a, dm, rad, dcdm, Hconf(a, dm, rad, dcdm, fourpiG, cosmo) / Hconf(1., dm, rad, dcdm, fourpiG, cosmo), scalarFT(kFT).real(), T00hom);
				fclose(outfile);
			}
		}
		// done recording background data
		
		prepareFTsource<Real>(phi, Sij, Sij, 2. * fourpiG * dx * dx / a);  // prepare nonlinear source for additional equations

	
		plan_Sij.execute(FFT_FORWARD);  // go to k-space


	
		projectFTscalar(SijFT, scalarFT);  // construct chi by scalar projection (k-space)

		
		plan_chi.execute(FFT_BACKWARD);	 // go back to position space

		chi.updateHalo();  // communicate halo values

		if (sim.vector_flag == VECTOR_ELLIPTIC)
		{

			plan_Bi.execute(FFT_FORWARD);

			projectFTvector(BiFT, BiFT, fourpiG * dx * dx); // solve B using elliptic constraint (k-space)

		}
		else
			evolveFTvector(SijFT, BiFT, a * a * dtau_old);  // evolve B using vector projection (k-space)

		if (sim.gr_flag > 0)
		{			
			plan_Bi.execute(FFT_BACKWARD);  // go back to position space

			Bi.updateHalo();  // communicate halo values
		}


		// lightcone output




// snapshot output
		// snapshot output
		{
			COUT << COLORTEXT_CYAN << " writing snapshot" << COLORTEXT_RESET << " at z = " << ((1./a) - 1.) <<  " (cycle " << cycle << "), tau/boxsize = " << tau << endl;

			writeSnapshots(sim, cosmo, fourpiG, a, dm, rad, dcdm, dtau_old, done_hij, snapcount, h5filename + sim.basename_snapshot, &pcls_cdm, &pcls_b, pcls_ncdm, &phi, &chi, &Bi, &source, &Sij, &scalarFT, &BiFT, &SijFT, &plan_phi, &plan_chi, &plan_Bi, &plan_source, &plan_Sij
#ifdef CHECK_B
				, &Bi_check, &BiFT_check, &plan_Bi_check
#endif
#ifdef VELOCITY
				, &vi
#endif
			);

			snapcount++;
		}

//powerspectra for each timestep		
                {
                        COUT << COLORTEXT_CYAN << " writing power spectra" << COLORTEXT_RESET << " at z = " << ((1./a) - 1.) <<  " (cycle " << cycle << "), tau/boxsize = " << tau << endl;

                        writeSpectra(sim, cosmo, fourpiG, a, pkcount,
#ifdef HAVE_CLASS
                                class_background, class_perturbs, ic,
#endif
                                &pcls_cdm, &pcls_b, pcls_ncdm, &phi, &chi, &Bi, &source, &Sij, &scalarFT, &BiFT, &SijFT, &plan_phi, &plan_chi, &plan_Bi, &plan_source, &plan_Sij
#ifdef CHECK_B
                                , &Bi_check, &BiFT_check, &plan_Bi_check
#endif
#ifdef VELOCITY
                                , &vi, &viFT, &plan_vi
#endif
                        );

                        pkcount++;
                }

//power spectra for normal time steps
		if (pkcount < sim.num_pk && 1. / a < sim.z_pk[pkcount] + 1.)
		{
			COUT << COLORTEXT_CYAN << " writing power spectra" << COLORTEXT_RESET << " at z = " << ((1./a) - 1.) <<  " (cycle " << cycle << "), tau/boxsize = " << tau << endl;

			writeSpectra(sim, cosmo, fourpiG, a, pkcount,
#ifdef HAVE_CLASS
				class_background, class_perturbs, ic,
#endif
				&pcls_cdm, &pcls_b, pcls_ncdm, &phi, &chi, &Bi, &source, &Sij, &scalarFT, &BiFT, &SijFT, &plan_phi, &plan_chi, &plan_Bi, &plan_source, &plan_Sij
#ifdef CHECK_B
				, &Bi_check, &BiFT_check, &plan_Bi_check
#endif
#ifdef VELOCITY
				, &vi, &viFT, &plan_vi
#endif
			);

			pkcount++;
		}
// // EXACT_OUTPUT_REDSHIFTS
//
//new stopping condition: z_out!

		if (1/a -1  <= sim.z_out) break;
		
//
//
		
//put back particle update

		// cdm and baryon particle update
		f_params[0] = a;
		f_params[1] = a * a * sim.numpts;
		if (sim.gr_flag > 0)
		{
			maxvel[0] = pcls_cdm.updateVel(update_q, (dtau + dtau_old) / 2., update_cdm_fields, (1. / a < ic.z_relax + 1. ? 3 : 2), f_params);
			if (sim.baryon_flag)
				maxvel[1] = pcls_b.updateVel(update_q, (dtau + dtau_old) / 2., update_b_fields, (1. / a < ic.z_relax + 1. ? 3 : 2), f_params);
		}
		else
		{
			maxvel[0] = pcls_cdm.updateVel(update_q_Newton, (dtau + dtau_old) / 2., update_cdm_fields, ((sim.radiation_flag + sim.fluid_flag > 0 && a < 1. / (sim.z_switch_linearchi + 1.)) ? 2 : 1), f_params);
			if (sim.baryon_flag)
				maxvel[1] = pcls_b.updateVel(update_q_Newton, (dtau + dtau_old) / 2., update_b_fields, ((sim.radiation_flag + sim.fluid_flag > 0 && a < 1. / (sim.z_switch_linearchi + 1.)) ? 2 : 1), f_params);
		}


				
		rungekutta4bg(a, dm, rad, dcdm, fourpiG, cosmo, 0.5 * dtau);  // evolve background by half a time step

		f_params[0] = a;
		f_params[1] = a * a * sim.numpts;
		if (sim.gr_flag > 0)
		{
			pcls_cdm.moveParticles(update_pos, dtau, update_cdm_fields, (1. / a < ic.z_relax + 1. ? 3 : 0), f_params);
			COUT << COLORTEXT_CYAN << " particles moving!!!!" << endl;	
			if (sim.baryon_flag)
				pcls_b.moveParticles(update_pos, dtau, update_b_fields, (1. / a < ic.z_relax + 1. ? 3 : 0), f_params);
			
		}
		else
		{ COUT << COLORTEXT_CYAN << " particles moving!!!!" << endl;	
			pcls_cdm.moveParticles(update_pos_Newton, dtau, NULL, 0, f_params);
			if (sim.baryon_flag)
				pcls_b.moveParticles(update_pos_Newton, dtau, NULL, 0, f_params);
		}


				
//
//background evolution

		rungekutta4bg(a, dm, rad, dcdm, fourpiG, cosmo, 0.5 * dtau);  // evolve background by half a time step
		COUT << COLORTEXT_CYAN << " dtau for this step is" << dtau  << endl;
		COUT << COLORTEXT_CYAN << "Hconf for this step is" << Hconf(a, dm, rad, dcdm, fourpiG, cosmo) << endl;
		COUT << COLORTEXT_CYAN << "tau for this step is" << tau  << endl;
		COUT << COLORTEXT_CYAN << "a for this step is" << a << endl;
		COUT << " total density of components = " <<  dm+rad+dcdm+cosmo.Omega_Lambda << endl;
		COUT << " Omega_m " <<  cosmo.Omega_m << endl;
		COUT << " dm + dcdm" <<  dm +dcdm << endl;
		COUT << " Omega_rad " <<  rad << endl;
//
		// done particle update
		
		tau += dtau;
		
		
		dtau_old = dtau;
		
		if (sim.Cf * dx < sim.steplimit / Hconf(a, dm, rad, dcdm, fourpiG, cosmo))
			dtau = sim.Cf * dx;
		else
			dtau = sim.steplimit / Hconf(a, dm, rad, dcdm, fourpiG, cosmo);
		   
		cycle++;
		
	}
	
	COUT << COLORTEXT_GREEN << " simulation complete." << COLORTEXT_RESET << endl;




#ifdef EXTERNAL_IO	
		ioserver.stop();
	}
#endif

	return 0;
}

