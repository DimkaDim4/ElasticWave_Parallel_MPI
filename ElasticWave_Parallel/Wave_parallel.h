#include "mpi.h"
#include <omp.h>
#include <iostream>
#include <functional>
#include <fstream>
#include <string>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stddef.h>

struct Vars
{
	double u_x;
	double u_y;
	double u_z;
	double sigma_xx;
	double sigma_yy;
	double sigma_zz;
	double sigma_xy;
	double sigma_yz;
	double sigma_zx;

	Vars & operator+=(const Vars& b)
	{
		u_x += b.u_x;
		u_y += b.u_y;
		u_z += b.u_z;
		sigma_xx += b.sigma_xx;
		sigma_yy += b.sigma_yy;
		sigma_zz += b.sigma_zz;
		sigma_xy += b.sigma_xy;
		sigma_yz += b.sigma_yz;
		sigma_zx += b.sigma_zx;
		return *this;
	}

	Vars operator /(const double& b)
	{
		u_x /= b;
		u_y /= b;
		u_z /= b;
		sigma_xx /= b;
		sigma_yy /= b;
		sigma_zz /= b;
		sigma_xy /= b;
		sigma_yz /= b;
		sigma_zx /= b;
		return *this;
	}

	friend Vars operator +(const Vars& rhs, const Vars& lhs)
	{
		Vars ret(rhs);
		return ret += lhs;
	}
};

class Wave3d
{
private:
	// параметры среды
	// v_p - скорость продольных волн
	// v_s - скорость поперечных волн
	// Rho - плотность
	double* v_p = nullptr;
	double* v_s = nullptr;
	double* Rho = nullptr;

	// Переменные, используемые для параллельной работы программы
	int ierror;
	int dims[2] = { 0, 0 };
	int periods[2] = { false, false };
	int reorder = false;
	int rank_coords[2];
	int size, rank;
	int left_rank, right_rank, lower_rank, upper_rank;

	MPI_Datatype mpi_vars;
	MPI_Datatype subarray_rw;
	int count_rw;
	double* buf_rw;

	MPI_Datatype left_subaray_send;
	MPI_Datatype left_subaray_recv;
	MPI_Datatype right_subaray_send;
	MPI_Datatype right_subaray_recv;
	MPI_Datatype upper_subaray_send;
	MPI_Datatype upper_subaray_recv;
	MPI_Datatype lower_subaray_send;
	MPI_Datatype lower_subaray_recv;

	int count_sendrecv_i = 0, count_sendrecv_j = 0;

	int starti_rw, endi_rw,
		startj_rw, endj_rw;

	int starts_f[3];

	// переменные, используемые для записи данных на файлы, не превышающих заданный размер
	long int max_size_file = 200 * 1024 * 1024; // размер файла в байтах // 200 Мбайт
	int num_file;

	int num_current_file_sigma = 1;
	int num_current_file_sigma_okt = 1;
	int num_current_file_abs_u = 1;
	int num_current_file_u_x = 1;
	int num_current_file_u_y = 1;
	int num_current_file_u_z = 1;

	int num_iters_in_file;

	int current_iter_sigma = 1;
	int current_iter_sigma_okt = 1;
	int current_iter_abs_u = 1;
	int current_iter_u_x = 1;
	int current_iter_u_y = 1;
	int current_iter_u_z = 1;

	MPI_File datafile_sigma;
	MPI_File datafile_sigma_okt;
	MPI_File datafile_abs_u;
	MPI_File datafile_u_x;
	MPI_File datafile_u_y;
	MPI_File datafile_u_z;

	MPI_Comm communicator;
	MPI_Status status;

	// параметры для численного решения

	double tau; // шаг по времени для решения системы уравнений переноса, расчет n+1 временного слоя с шагом 3*tau
	double h; // шаг по регулярной сетки
	double T; // время расчета
	int I; // количество узлов в каждом направлении
	int N; // количество итераций

	long int _size_i;
	long int _size_j;
	long int _size_k;

	long int _I;
	long int _J;

	int PML_Size = 15; // толщина PML-слоя
	double sigma_max = 40;

	// источник волновых возмущений - импульс Рикера
	std::function<double(double, double, double, double)> f;
	//int i_f, j_f, k_f;

	Vars* w_rank_curr;
	Vars* w_rank_next;
	Vars* g_rank_curr;
	Vars* g_rank_next;

	Vars* w_curr;
	Vars* w_next;

	// разностная схема для решения уравнения переноса
	int scheme_type;

	// внешний источник
	void F(const int& n);

	void _change_1(const long int& index);
	void _change_2(const long int& index);
	void _change_3(const long int& index);

	void _back_change_1(const long int& index);
	void _back_change_2(const long int& index);
	void _back_change_3(const long int& index);

	void _transfer_eq_x(const long int& index);
	void _transfer_eq_y(const long int& index);
	void _transfer_eq_z(const long int& index);

	void _PML_transfer_eq_x(const long int& index, const double& demp);
	void _PML_transfer_eq_y(const long int& index, const double& demp);
	void _PML_transfer_eq_z(const long int& index, const double& demp);

	void _solve_system_eq(const long int& index, const Vars& umm, const Vars& um, const Vars& u, const Vars& up, const Vars& upp);
	void _PML_solve_system_eq(const long int& index, const Vars& umm, const Vars& um, const Vars& u, const Vars& up, const Vars& upp, const double& demp);

	double _get_alpha(const double& sigma, const double& _deltap, const double& _deltam);
	double _PML_get_alpha(const double& sigma, const double& _deltap, const double& _deltam);

	void _make_step_X();
	void _make_step_Y();
	void _make_step_Z();

	void _write_to_file_Sigma();
	void _write_to_file_Sigma_Okt();
	void _write_to_file_Abs_U();
	void _write_to_file_Ux();
	void _write_to_file_Uy();
	void _write_to_file_Uz();

	void _read_param_from_file();

public:
	Wave3d(int I, double T, std::function<double(double, double, double, double)> f);
	~Wave3d();
	void solve();
};