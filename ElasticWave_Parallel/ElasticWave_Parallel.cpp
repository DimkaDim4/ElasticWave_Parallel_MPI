#include "Wave_parallel.h"

double PI = 3.141592653589793;
double v0 = 30.;
double t0 = 1. / v0;
int I = 50;
double h = 1. / (I - 1.);
double A = 1000.;
double T = 0.3;

double f(double t, double x, double y, double z)
{
    if ((fabs(x - 0.5) <= h) && (fabs(y - 0.5) <= h) && (fabs(z - 0.2) <= h) && (t <= 2. * t0))
        return A * 2. * PI * v0 * sqrt(exp(1.)) * (t0 - t) * exp(-2. * PI * PI * v0 * v0 * (t - t0) * (t - t0));
    return 0.;
}

int main(int argc, char *argv[])
{
    int root = 0;
    int MyID, NumProc, ierror;
    MPI_Status status;

    ierror = MPI_Init(&argc, &argv);
    if (ierror != MPI_SUCCESS)
    {
        printf("MPI initialization error!");
        exit(1);
    }

    MPI_Comm_size(MPI_COMM_WORLD, &NumProc);
    MPI_Comm_rank(MPI_COMM_WORLD, &MyID);

    if (MyID == 0)
    {
        std::ofstream datafileVp("Vp.bin", std::ios::binary | std::ios::out);
        std::ofstream datafileVs("Vs.bin", std::ios::binary | std::ios::out);
        std::ofstream datafileRho("Rho.bin", std::ios::binary | std::ios::out);

        double vp, vs, rho;

        for (int i = 0; i < I; i++)
        {
            for (int j = 0; j < I; j++)
            {
                for (int k = 0; k < I / 3; k++)
                {
                    vp = 3.2 * 2.0;
                    vs = 1.82 * 2.0;
                    rho = 2.7 * 2.0;

                    datafileVp.write((char*)&vp, sizeof(double));
                    datafileVs.write((char*)&vs, sizeof(double));
                    datafileRho.write((char*)&rho, sizeof(double));
                }
                for (int k = I / 3; k < 2 * I / 3; k++)
                {
                    vp = 5.9 * 2.0;
                    vs = 3.42 * 2.0;
                    rho = 2.85 * 2.0;

                    datafileVp.write((char*)&vp, sizeof(double));
                    datafileVs.write((char*)&vs, sizeof(double));
                    datafileRho.write((char*)&rho, sizeof(double));
                }
                for (int k = 2 * I / 3; k < I; k++)
                {
                    vp = 6.95 * 2.0;
                    vs = 4.03 * 2.0;
                    rho = 2.81 * 2.0;

                    datafileVp.write((char*)&vp, sizeof(double));
                    datafileVs.write((char*)&vs, sizeof(double));
                    datafileRho.write((char*)&rho, sizeof(double));
                }
            }
        }
        datafileVp.close();
        datafileVs.close();
        datafileRho.close();
    }
    
    double tstart = MPI_Wtime();
    Wave3d wave(I, T, f);
    MPI_Barrier(MPI_COMM_WORLD);

    wave.solve();

    MPI_Barrier(MPI_COMM_WORLD);
    if (MyID == root)
        printf("Time: %lf\n", MPI_Wtime() - tstart);

    return 0;
}