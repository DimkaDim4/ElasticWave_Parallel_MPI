#include "Wave_parallel.h"

void Wave3d::F()
{
    for (int i = starti_rw; i < endi_rw; ++i)
    {
        for (int j = startj_rw; j < endj_rw; ++j)
        {
            for (int k = 2 + PML_Size; k < 2 + I + PML_Size; ++k)
            {
                long int index = i * _I + j * _J + k;
                double _f = f(n * tau * 3., (starts_f[0] + i - starti_rw) * h, (starts_f[1] + j - startj_rw) * h, (k - 2 - PML_Size) * h) * tau * 3;
                //double _f = f(n * tau * 3., (i - starti_rw) * h, (j - startj_rw) * h, (PML_Size) * h) * tau * 3;
                if (_f != 0.)
                {
                    //printf("rank = %d f: (%d, %d, %d) f = %f\n", rank, starts_f[0] + i - starti_rw, starts_f[1] + j - startj_rw, k - 2 - PML_Size, _f);
                    //w_rank_curr[index].u_x += _f;
                    //w_rank_curr[index].u_y += _f;
                    //w_rank_curr[index].u_z += _f;

                    w_rank_curr[index].sigma_xx += _f;
                    w_rank_curr[index].sigma_yy += _f;
                    w_rank_curr[index].sigma_zz += _f;

                    //w_rank_curr[index].sigma_xy += _f;
                    //w_rank_curr[index].sigma_yz += _f;
                    //w_rank_curr[index].sigma_zx += _f;
                }
            }
        }
    }
}

void Wave3d::_change_1(const long int& index)
{
    double vs = v_s[index];
    double vp = v_p[index];
    double rho = Rho[index];
    double mu = rho * (vs * vs);
    double lam = rho * (vp * vp) - 2. * mu;

    Vars g;
    Vars w = w_rank_curr[index];

    g.u_x = lam * 0.5 * (w.u_x / vp + w.sigma_xx / (lam + 2. * mu));
    g.u_y = 0.5 * (w.u_z * vs * rho + w.sigma_zx);
    g.u_z = 0.5 * (w.u_y * vs * rho + w.sigma_xy);

    g.sigma_xx = w.sigma_yz;
    g.sigma_yy = w.sigma_zz - w.sigma_xx * lam / (lam + 2. * mu);
    g.sigma_zz = w.sigma_yy - w.sigma_xx * lam / (lam + 2. * mu);

    g.sigma_xy = 0.5 * (-w.u_z * vs * rho + w.sigma_zx);
    g.sigma_yz = 0.5 * (-w.u_y * vs * rho + w.sigma_xy);
    g.sigma_zx = lam * 0.5 * (-w.u_x / vp + w.sigma_xx / (lam + 2. * mu));

    g_rank_curr[index] = g;
}

void Wave3d::_change_2(const long int& index)
{
    double vs = v_s[index];
    double vp = v_p[index];
    double rho = Rho[index];
    double mu = rho * (vs * vs);
    double lam = rho * (vp * vp) - 2. * mu;

    Vars g;
    Vars w = w_rank_curr[index];

    g.u_x = lam * 0.5 * (w.u_y / vp + w.sigma_yy / (lam + 2. * mu));
    g.u_y = 0.5 * (w.u_z * vs * rho + w.sigma_yz);
    g.u_z = 0.5 * (w.u_x * vs * rho + w.sigma_xy);

    g.sigma_xx = w.sigma_xx - w.sigma_yy * lam / (lam + 2. * mu);
    g.sigma_yy = w.sigma_zx;
    g.sigma_zz = w.sigma_zz - w.sigma_yy * lam / (lam + 2. * mu);

    g.sigma_xy = 0.5 * (-w.u_z * vs * rho + w.sigma_yz);
    g.sigma_yz = 0.5 * (-w.u_x * vs * rho + w.sigma_xy);
    g.sigma_zx = lam * 0.5 * (-w.u_y / vp + w.sigma_yy / (lam + 2. * mu));

    g_rank_curr[index] = g;
}

void Wave3d::_change_3(const long int& index)
{
    double vs = v_s[index];
    double vp = v_p[index];
    double rho = Rho[index];
    double mu = rho * (vs * vs);
    double lam = rho * (vp * vp) - 2. * mu;

    Vars g;
    Vars w = w_rank_curr[index];

    g.u_x = 0.5 * (w.u_z * vp * rho + w.sigma_zz);
    g.u_y = 0.5 * (w.u_y * vs * rho + w.sigma_yz);
    g.u_z = 0.5 * (w.u_x * vs * rho + w.sigma_zx);

    g.sigma_xx = w.sigma_xy;
    g.sigma_yy = w.sigma_yy - w.sigma_zz * lam / (lam + 2. * mu);
    g.sigma_zz = w.sigma_xx - w.sigma_zz * lam / (lam + 2. * mu);

    g.sigma_xy = 0.5 * (-w.u_x * vs * rho + w.sigma_zx);
    g.sigma_yz = 0.5 * (-w.u_y * vs * rho + w.sigma_yz);
    g.sigma_zx = 0.5 * (-w.u_z * vp * rho + w.sigma_zz);

    g_rank_curr[index] = g;
}

void Wave3d::_back_change_1(const long int& index)
{
    double vs = v_s[index];
    double vp = v_p[index];
    double rho = Rho[index];
    double mu = rho * (vs * vs);
    double lam = rho * (vp * vp) - 2. * mu;

    Vars w;
    Vars g = g_rank_next[index];

    w.u_x = vp / lam * (g.u_x - g.sigma_zx);
    w.u_y = (g.u_z - g.sigma_yz) / (vs * rho);
    w.u_z = (g.u_y - g.sigma_xy) / (vs * rho);

    w.sigma_xx = (g.u_x + g.sigma_zx) * (lam + 2. * mu) / lam;
    w.sigma_yy = g.u_x + g.sigma_zz + g.sigma_zx;
    w.sigma_zz = g.u_x + g.sigma_yy + g.sigma_zx;

    w.sigma_xy = g.u_z + g.sigma_yz;
    w.sigma_yz = g.sigma_xx;
    w.sigma_zx = g.u_y + g.sigma_xy;

    w_rank_next[index] = w;
}

void Wave3d::_back_change_2(const long int& index)
{
    double vs = v_s[index];
    double vp = v_p[index];
    double rho = Rho[index];
    double mu = rho * (vs * vs);
    double lam = rho * (vp * vp) - 2. * mu;

    Vars w;
    Vars g = g_rank_next[index];

    w.u_x = (g.u_z - g.sigma_yz) / (vs * rho);
    w.u_y = vp / lam * (g.u_x - g.sigma_zx);
    w.u_z = (g.u_y - g.sigma_xy) / (vs * rho);

    w.sigma_xx = g.u_x + g.sigma_xx + g.sigma_zx;
    w.sigma_yy = (g.u_x + g.sigma_zx) * (lam + 2. * mu) / lam;
    w.sigma_zz = g.u_x + g.sigma_zz + g.sigma_zx;

    w.sigma_xy = g.u_z + g.sigma_yz;
    w.sigma_yz = g.u_y + g.sigma_xy;
    w.sigma_zx = g.sigma_yy;

    w_rank_next[index] = w;
}

void Wave3d::_back_change_3(const long int& index)
{
    double vs = v_s[index];
    double vp = v_p[index];
    double rho = Rho[index];
    double mu = rho * (vs * vs);
    double lam = rho * (vp * vp) - 2. * mu;

    Vars w;
    Vars g = g_rank_next[index];

    w.u_x = (g.u_z - g.sigma_xy) / (vs * rho);
    w.u_y = (g.u_y - g.sigma_yz) / (vs * rho);
    w.u_z = (g.u_x - g.sigma_zx) / (vp * rho);

    w.sigma_xx = (g.u_x + g.sigma_zx) * lam / (lam + 2. * mu) + g.sigma_zz;
    w.sigma_yy = (g.u_x + g.sigma_zx) * lam / (lam + 2. * mu) + g.sigma_yy;
    w.sigma_zz = g.u_x + g.sigma_zx;

    w.sigma_xy = g.sigma_xx;
    w.sigma_yz = g.u_y + g.sigma_yz;
    w.sigma_zx = g.u_z + g.sigma_xy;

    w_rank_next[index] = w;
}

void Wave3d::_transfer_eq_x(const long int& index)
{
    this->_solve_system_eq(index, g_rank_curr[index - 2 * _I], g_rank_curr[index - _I], g_rank_curr[index], g_rank_curr[index + _I], g_rank_curr[index + 2 * _I]);
}

void Wave3d::_transfer_eq_y(const long int& index)
{
    this->_solve_system_eq(index, g_rank_curr[index - 2 * _J], g_rank_curr[index - _J], g_rank_curr[index], g_rank_curr[index + _J], g_rank_curr[index + 2 * _J]);
}

void Wave3d::_transfer_eq_z(const long int& index)
{
    this->_solve_system_eq(index, g_rank_curr[index - 2], g_rank_curr[index - 1], g_rank_curr[index], g_rank_curr[index + 1], g_rank_curr[index + 2]);
}

void Wave3d::_PML_transfer_eq_x(const long int& index, const double& demp)
{
    this->_PML_solve_system_eq(index, g_rank_curr[index - 2 * _I], g_rank_curr[index - _I], g_rank_curr[index], g_rank_curr[index + _I], g_rank_curr[index + 2 * _I], demp);
}

void Wave3d::_PML_transfer_eq_y(const long int& index, const double& demp)
{
    this->_PML_solve_system_eq(index, g_rank_curr[index - 2 * _J], g_rank_curr[index - _J], g_rank_curr[index], g_rank_curr[index + _J], g_rank_curr[index + 2 * _J], demp);
}

void Wave3d::_PML_transfer_eq_z(const long int& index, const double& demp)
{
    this->_PML_solve_system_eq(index, g_rank_curr[index - 2], g_rank_curr[index - 1], g_rank_curr[index], g_rank_curr[index + 1], g_rank_curr[index + 2], demp);
}

void Wave3d::_solve_system_eq(const long int& index, const Vars& umm, const Vars& um, const Vars& u, const Vars& up, const Vars& upp)
{
    //double vp = 2. / (1. / v_p[index] + 1. / v_p[index + _I]);
    //double vs = 2. / (1. / v_s[index] + 1. / v_s[index + _I]);
    double vp = v_p[index];
    double vs = v_s[index];

    double sigma; // ����� ������� 
    double deltam; // delta -1
    double delta0; // delta 0
    double deltap; // delta +1
    double alpha;

    // 1-e ��������� ��������
    // lambda = -vp
    sigma = vp * tau / h;
    deltam = upp.u_x - up.u_x; // delta -1
    delta0 = up.u_x - u.u_x; // delta 0
    deltap = u.u_x - um.u_x; // delta +1

    if (delta0 != 0)
    {
        alpha = this->_get_alpha(sigma, deltap / delta0, deltam / delta0);
        g_rank_next[index].u_x = g_rank_curr[index].u_x + 0.5 * sigma * (delta0 + deltap) + 0.5 * sigma * sigma * (delta0 - deltap) + alpha * (deltam - 2. * delta0 + deltap);
    }
    else
        g_rank_next[index].u_x = g_rank_curr[index].u_x;

    // 2-e ��������� ��������
    // lambda = -vs
    sigma = vs * tau / h;
    deltam = upp.u_y - up.u_y; // delta -1
    delta0 = up.u_y - u.u_y; // delta 0
    deltap = u.u_y - um.u_y; // delta +1

    if (delta0 != 0)
    {
        alpha = this->_get_alpha(sigma, deltap / delta0, deltam / delta0);
        g_rank_next[index].u_y = g_rank_curr[index].u_y + 0.5 * sigma * (delta0 + deltap) + 0.5 * sigma * sigma * (delta0 - deltap) + alpha * (deltam - 2. * delta0 + deltap);
    }
    else
        g_rank_next[index].u_y = g_rank_curr[index].u_y;

    // 3-e ��������� ��������
    // lambda = -vs
    //sigma = vs * tau / h;
    deltam = upp.u_z - up.u_z; // delta -1
    delta0 = up.u_z - u.u_z; // delta 0
    deltap = u.u_z - um.u_z; // delta +1

    if (delta0 != 0)
    {
        alpha = this->_get_alpha(sigma, deltap / delta0, deltam / delta0);
        g_rank_next[index].u_z = g_rank_curr[index].u_z + 0.5 * sigma * (delta0 + deltap) + 0.5 * sigma * sigma * (delta0 - deltap) + alpha * (deltam - 2. * delta0 + deltap);
    }
    else
        g_rank_next[index].u_z = g_rank_curr[index].u_z;

    // 4-e ��������� ��������
    // lambda = 0
    g_rank_next[index].sigma_xx = g_rank_curr[index].sigma_xx;

    // 5-e ��������� ��������
    // lambda = 0
    g_rank_next[index].sigma_yy = g_rank_curr[index].sigma_yy;

    // 6-e ��������� ��������
    // lambda = 0
    g_rank_next[index].sigma_zz = g_rank_curr[index].sigma_zz;


    //vp = 4. / (1. / v_p[index - 2 * _I] + 1. / v_p[index - _I] + 1. / v_p[index] + 1. / v_p[index + _I]);
    //vs = 4. / (1. / v_s[index - 2 * _I] + 1. / v_s[index - _I] + 1. / v_s[index] + 1. / v_s[index + _I]);
    vp = 2. / (1. / v_p[index] + 1. / v_p[index - _I]);
    vs = 2. / (1. / v_s[index] + 1. / v_s[index - _I]);
    // double vp = v_p[index];
    // double vs = v_s[index];

    // 7e ��������� ��������
    // lambda = vs
    sigma = vs * tau / h;
    deltam = umm.sigma_xy - um.sigma_xy; // delta -1
    delta0 = um.sigma_xy - u.sigma_xy; // delta 0
    deltap = u.sigma_xy - up.sigma_xy; // delta +1

    if (delta0 != 0)
    {
        alpha = this->_get_alpha(sigma, deltap / delta0, deltam / delta0);
        g_rank_next[index].sigma_xy = g_rank_curr[index].sigma_xy + 0.5 * sigma * (delta0 + deltap) + 0.5 * sigma * sigma * (delta0 - deltap) + alpha * (deltam - 2. * delta0 + deltap);
    }
    else
        g_rank_next[index].sigma_xy = g_rank_curr[index].sigma_xy;

    // 8e ��������� ��������
    // lambda = vs
    //sigma = vs * tau / h;
    deltam = umm.sigma_yz - um.sigma_yz; // delta -1
    delta0 = um.sigma_yz - u.sigma_yz; // delta 0
    deltap = u.sigma_yz - up.sigma_yz; // delta +1

    if (delta0 != 0)
    {
        alpha = this->_get_alpha(sigma, deltap / delta0, deltam / delta0);
        g_rank_next[index].sigma_yz = g_rank_curr[index].sigma_yz + 0.5 * sigma * (delta0 + deltap) + 0.5 * sigma * sigma * (delta0 - deltap) + alpha * (deltam - 2. * delta0 + deltap);
    }
    else
        g_rank_next[index].sigma_yz = g_rank_curr[index].sigma_yz;

    // 9e ��������� ��������
    // lambda = vp
    sigma = vp * tau / h;
    deltam = umm.sigma_zx - um.sigma_zx; // delta -1
    delta0 = um.sigma_zx - u.sigma_zx; // delta 0
    deltap = u.sigma_zx - up.sigma_zx; // delta +1

    if (delta0 != 0)
    {
        alpha = this->_get_alpha(sigma, deltap / delta0, deltam / delta0);
        g_rank_next[index].sigma_zx = g_rank_curr[index].sigma_zx + 0.5 * sigma * (delta0 + deltap) + 0.5 * sigma * sigma * (delta0 - deltap) + alpha * (deltam - 2. * delta0 + deltap);
    }
    else
        g_rank_next[index].sigma_zx = g_rank_curr[index].sigma_zx;
}

void Wave3d::_PML_solve_system_eq(const long int& index, const Vars& umm, const Vars& um, const Vars& u, const Vars& up, const Vars& upp, const double& demp)
{
    //double vp = 2. / (1. / v_p[index] + 1. / v_p[index + _I]);
    //double vs = 2. / (1. / v_s[index] + 1. / v_s[index + _I]);
    double vp = v_p[index];
    double vs = v_s[index];

    double sigma; // ����� ������� 
    double deltam; // delta -1
    double delta0; // delta 0
    double deltap; // delta +1
    double alpha;

    double coeff = 1. - tau * demp;

    // 1-e ��������� ��������
    // lambda = -vp
    sigma = vp * tau / h;
    deltam = upp.u_x - up.u_x; // delta -1
    delta0 = up.u_x - u.u_x; // delta 0
    deltap = u.u_x - um.u_x; // delta +1

    if (delta0 != 0)
    {
        alpha = this->_get_alpha(sigma, deltap / delta0, deltam / delta0);
        switch (this->scheme_type)
        {
        case 0: // ����� �����-���������
            g_rank_next[index].u_x = u.u_x * coeff
                + 0.5 * sigma * (delta0 * coeff + deltap) + 0.5 * sigma * sigma * (delta0 - deltap) + alpha * (deltam - 2. * delta0 + deltap);
            break;
        case 1: // ����� ����-��������
            g_rank_next[index].u_x = u.u_x * (1. - 1.5 * tau * demp)
                + 0.5 * sigma * (delta0 * (1. - 2. * tau * demp) + deltap) + 0.5 * sigma * sigma * (delta0 - deltap) + alpha * (deltam - 2. * delta0 + deltap);
            break;
        case 2: // ����� �������
            g_rank_next[index].u_x = u.u_x * coeff
                + 0.5 * sigma * (delta0 + deltap) + 0.5 * sigma * sigma * (delta0 - deltap) + alpha * (deltam - 2. * delta0 + deltap)
                - 0.25 * sigma * tau * demp * (delta0 + deltap) - tau * sigma * sigma * demp * (deltam - deltap) / 12.;
            break;
        }
    }
    else
        g_rank_next[index].u_x = u.u_x * coeff;

    // 2-e ��������� ��������
    // lambda = -vs
    sigma = vs * tau / h;
    deltam = upp.u_y - up.u_y; // delta -1
    delta0 = up.u_y - u.u_y; // delta 0
    deltap = u.u_y - um.u_y; // delta +1

    if (delta0 != 0)
    {
        alpha = this->_get_alpha(sigma, deltap / delta0, deltam / delta0);
        switch (this->scheme_type)
        {
        case 0: // ����� �����-���������
            g_rank_next[index].u_y = u.u_y * coeff
                + 0.5 * sigma * (delta0 * coeff + deltap) + 0.5 * sigma * sigma * (delta0 - deltap) + alpha * (deltam - 2. * delta0 + deltap);
            break;
        case 1: // ����� ����-��������
            g_rank_next[index].u_y = u.u_y * (1. - 1.5 * tau * demp)
                + 0.5 * sigma * (delta0 * (1. - 2. * tau * demp) + deltap) + 0.5 * sigma * sigma * (delta0 - deltap) + alpha * (deltam - 2. * delta0 + deltap);
            break;
        case 2: // ����� �������
            g_rank_next[index].u_y = u.u_y * coeff
                + 0.5 * sigma * (delta0 + deltap) + 0.5 * sigma * sigma * (delta0 - deltap) + alpha * (deltam - 2. * delta0 + deltap)
                - 0.25 * sigma * tau * demp * (delta0 + deltap) - tau * sigma * sigma * demp * (deltam - deltap) / 12.;
            break;
        }
    }
    else
        g_rank_next[index].u_y = u.u_y * coeff;

    // 3-e ��������� ��������
    // lambda = -vs
    //sigma = vs * tau / h;
    deltam = upp.u_z - up.u_z; // delta -1
    delta0 = up.u_z - u.u_z; // delta 0
    deltap = u.u_z - um.u_z; // delta +1

    if (delta0 != 0)
    {
        alpha = this->_get_alpha(sigma, deltap / delta0, deltam / delta0);
        switch (this->scheme_type)
        {
        case 0: // ����� �����-���������
            g_rank_next[index].u_z = u.u_z * coeff
                + 0.5 * sigma * (delta0 * coeff + deltap) + 0.5 * sigma * sigma * (delta0 - deltap) + alpha * (deltam - 2. * delta0 + deltap);
            break;
        case 1: // ����� ����-��������
            g_rank_next[index].u_z = u.u_z * (1. - 1.5 * tau * demp)
                + 0.5 * sigma * (delta0 * (1. - 2. * tau * demp) + deltap) + 0.5 * sigma * sigma * (delta0 - deltap) + alpha * (deltam - 2. * delta0 + deltap);
            break;
        case 2: // ����� �������
            g_rank_next[index].u_z = u.u_z * coeff
                + 0.5 * sigma * (delta0 + deltap) + 0.5 * sigma * sigma * (delta0 - deltap) + alpha * (deltam - 2. * delta0 + deltap)
                - 0.25 * sigma * tau * demp * (delta0 + deltap) - tau * sigma * sigma * demp * (deltam - deltap) / 12.;
            break;
        }
    }
    else
        g_rank_next[index].u_z = u.u_z * coeff;

    // 4-e ��������� ��������
    // lambda = 0
    g_rank_next[index].sigma_xx = u.sigma_xx * coeff;

    // 5-e ��������� ��������
    // lambda = 0
    g_rank_next[index].sigma_yy = u.sigma_yy * coeff;

    // 6-e ��������� ��������
    // lambda = 0
    g_rank_next[index].sigma_zz = u.sigma_zz * coeff;


    //vp = 4. / (1. / v_p[index - 2 * _I] + 1. / v_p[index - _I] + 1. / v_p[index] + 1. / v_p[index + _I]);
    //vs = 4. / (1. / v_s[index - 2 * _I] + 1. / v_s[index - _I] + 1. / v_s[index] + 1. / v_s[index + _I]);
    //vp = 2. / (1. / v_p[index] + 1. / v_p[index - _I]);
    //vs = 2. / (1. / v_s[index] + 1. / v_s[index - _I]);
    // double vp = v_p[index];
    // double vs = v_s[index];

    // 7e ��������� ��������
    // lambda = vs
    sigma = vs * tau / h;
    deltam = umm.sigma_xy - um.sigma_xy; // delta -1
    delta0 = um.sigma_xy - u.sigma_xy; // delta 0
    deltap = u.sigma_xy - up.sigma_xy; // delta +1

    if (delta0 != 0)
    {
        alpha = this->_get_alpha(sigma, deltap / delta0, deltam / delta0);
        switch (this->scheme_type)
        {
        case 0: // ����� �����-���������
            g_rank_next[index].sigma_xy = u.sigma_xy * coeff
                + 0.5 * sigma * (delta0 * coeff + deltap) + 0.5 * sigma * sigma * (delta0 - deltap) + alpha * (deltam - 2. * delta0 + deltap);
            break;
        case 1: // ����� ����-��������
            g_rank_next[index].sigma_xy = u.sigma_xy * (1. - 1.5 * tau * demp)
                + 0.5 * sigma * (delta0 * (1. - 2. * tau * demp) + deltap) + 0.5 * sigma * sigma * (delta0 - deltap) + alpha * (deltam - 2. * delta0 + deltap);
            break;
        case 2: // ����� �������
            g_rank_next[index].sigma_xy = u.sigma_xy * coeff
                + 0.5 * sigma * (delta0 + deltap) + 0.5 * sigma * sigma * (delta0 - deltap) + alpha * (deltam - 2. * delta0 + deltap)
                - 0.25 * sigma * tau * demp * (delta0 + deltap) - tau * sigma * sigma * demp * (deltam - deltap) / 12.;
            break;
        }
    }
    else
        g_rank_next[index].sigma_xy = u.sigma_xy * coeff;

    // 8e ��������� ��������
    // lambda = vs
    //sigma = vs * tau / h;
    deltam = umm.sigma_yz - um.sigma_yz; // delta -1
    delta0 = um.sigma_yz - u.sigma_yz; // delta 0
    deltap = u.sigma_yz - up.sigma_yz; // delta +1

    if (delta0 != 0)
    {
        alpha = this->_get_alpha(sigma, deltap / delta0, deltam / delta0);
        switch (this->scheme_type)
        {
        case 0: // ����� �����-���������
            g_rank_next[index].sigma_yz = u.sigma_yz * coeff
                + 0.5 * sigma * (delta0 * coeff + deltap) + 0.5 * sigma * sigma * (delta0 - deltap) + alpha * (deltam - 2. * delta0 + deltap);
            break;
        case 1: // ����� ����-��������
            g_rank_next[index].sigma_yz = u.sigma_yz * (1. - 1.5 * tau * demp)
                + 0.5 * sigma * (delta0 * (1. - 2. * tau * demp) + deltap) + 0.5 * sigma * sigma * (delta0 - deltap) + alpha * (deltam - 2. * delta0 + deltap);
            break;
        case 2: // ����� �������
            g_rank_next[index].sigma_yz = u.sigma_yz * coeff
                + 0.5 * sigma * (delta0 + deltap) + 0.5 * sigma * sigma * (delta0 - deltap) + alpha * (deltam - 2. * delta0 + deltap)
                - 0.25 * sigma * tau * demp * (delta0 + deltap) - tau * sigma * sigma * demp * (deltam - deltap) / 12.;
            break;
        }
    }
    else
        g_rank_next[index].sigma_yz = u.sigma_yz * coeff;

    // 9e ��������� ��������
    // lambda = vp
    sigma = vp * tau / h;
    deltam = umm.sigma_zx - um.sigma_zx; // delta -1
    delta0 = um.sigma_zx - u.sigma_zx; // delta 0
    deltap = u.sigma_zx - up.sigma_zx; // delta +1

    if (delta0 != 0)
    {
        alpha = this->_get_alpha(sigma, deltap / delta0, deltam / delta0);
        switch (this->scheme_type)
        {
        case 0: // ����� �����-���������
            g_rank_next[index].sigma_zx = u.sigma_zx * coeff
                + 0.5 * sigma * (delta0 * coeff + deltap) + 0.5 * sigma * sigma * (delta0 - deltap) + alpha * (deltam - 2. * delta0 + deltap);
            break;
        case 1: // ����� ����-��������
            g_rank_next[index].sigma_zx = u.sigma_zx * (1. - 1.5 * tau * demp)
                + 0.5 * sigma * (delta0 * (1. - 2. * tau * demp) + deltap) + 0.5 * sigma * sigma * (delta0 - deltap) + alpha * (deltam - 2. * delta0 + deltap);
            break;
        case 2: // ����� �������
            g_rank_next[index].sigma_zx = u.sigma_zx * coeff
                + 0.5 * sigma * (delta0 + deltap) + 0.5 * sigma * sigma * (delta0 - deltap) + alpha * (deltam - 2. * delta0 + deltap)
                - 0.25 * sigma * tau * demp * (delta0 + deltap) - tau * sigma * sigma * demp * (deltam - deltap) / 12.;
            break;
        }
    }
    else
        g_rank_next[index].sigma_zx = u.sigma_zx * coeff;
}

double Wave3d::_get_alpha(const double& sigma, const double& _deltap, const double& _deltam)
{
    if (this->is_hybrid_scheme) // ��������� ����� (����������� ����������� "����������" ���������, �������� ���������� ������)
    {
        double coeff, w;
        w = sigma * (1. + sigma) * 0.5 + _deltap * sigma * (1. - sigma) * 0.5;
        if ((w >= 0.) && (w <= 1.))
        {
            this->scheme_type = 0;
            return 0.; // ����� �����-���������
        }

        w = sigma * (3. - sigma) * 0.5 + _deltam * sigma * (sigma - 1.) * 0.5;
        if ((w >= 0.) && (w <= 1.))
        {
            this->scheme_type = 1;
            return 0.5 * sigma * (sigma - 1.); // ����� ����-��������
        }

        coeff = sigma * (sigma * sigma - 1.) / 6.;
        w = 0.5 * sigma * (1. + _deltap) + 0.5 * sigma * sigma * (1. - _deltap) + coeff * (_deltam + _deltap - 2.);
        if ((w >= 0.) && (w <= 1.))
        {
            this->scheme_type = 2;
            return coeff; // ����� ����� ��������
        }
    }
    else if (this->is_LaxWendroff_scheme)
    {
        return 0.;
    }
    else if (this->is_BeamWarming_scheme)
    {
        return sigma * (sigma - 1.) / 2.;
    }
    else if (this->is_Rusanov_scheme)
    {
        return sigma * (sigma * sigma - 1.) / 6.;
    }

    this->scheme_type = 2;
    return sigma * (sigma * sigma - 1.) / 6.;
}

void Wave3d::_make_step_X()
{
    // ������� ���������
    // dW/dt + A * dW/dx = 0 

    long int index = 0;
    // ������� � ������������������ �������
    // (������� ������� � ����������� ������)
    for (int i = 0; i < _size_i; ++i)
    {
        for (int j = 2; j < _size_j - 2; ++j)
        {
            for (int k = 2; k < _size_k - 2; ++k)
            {
                index = i * _I + j * _J + k;
                _change_1(index);
            }
        }
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //      rank_coords[0] = 0                rank_coords[0] = 1                     rank_coords[0] = dims[0]-2        rank_coords[0] = dims[0]-1
    //      +----------------+                +----------------+                         +---------------+                +----------------+
    //      |                |                |                |                         |               |                |                |
    // +----+-----+-----+----+----+      +----+----+------+----+----+               +----+----+-----+----+----+      +----+----+-----+-----+----+
    // | BC | PML |     | ** | ++ |  ->  | ** | ++ |      | ** | ++ |  ->  ...  ->  | ** | ++ |     | ** | ++ |  ->  | ** | ++ |     | PML | BC |
    // | BC | PML |     | ** | ++ |      | ** | ++ |      | ** | ++ |      ...      | ** | ++ |     | ** | ++ |      | ** | ++ |     | PML | BC |
    // | BC | PML |     | ** | ++ |      | ** | ++ |      | ** | ++ |      ...      | ** | ++ |     | ** | ++ |      | ** | ++ |     | PML | BC |
    // | BC | PML |     | ** | ++ |  <-  | ** | ++ |      | ** | ++ |  <-  ...  <-  | ** | ++ |     | ** | ++ |  <-  | ** | ++ |     | PML | BC |
    // +----+-----+-----+----+----+      +----+----+------+----+----+               +----+----+-----+----+----+      +----+----+-----+-----+----+
    //      |                |                |                |                         |               |                |                |
    //      +----------------+                +----------------+                         +---------------+                +----------------+
    //
    // -> ����� ������������� ������ ����� ���������� ��� ������� ��������� dW/dt + A * dW/dx = 0
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    // ����� ������ �� ��� x. ������ ��� �������� ����� ���������������.
    // ���������� ������ "������" �����, ��������� ������ �� "������" ������.
    // � ������, ���� "������" ����� ���, �� lower_rank = MPI_PROC_NULL,
    // � ���� ������ ����� ������� MPI_Sendrecv ���������� ��� ������, �������� �������� �� ������������,
    // ����� ���������� ������ ����������.
    // ����������, ���� "�����" ������ �����������, ���������� ������ �������� ������ ������� ��������.
    MPI_Sendrecv(
        g_rank_curr, 1, lower_subaray_send, lower_rank, 200,
        g_rank_curr, 1, upper_subaray_recv, upper_rank, 200,
        communicator, MPI_STATUS_IGNORE);

    // ���������� ����������� ������ ������� MPI_Sendrecv
    MPI_Sendrecv(
        g_rank_curr, 1, upper_subaray_send, upper_rank, 201,
        g_rank_curr, 1, lower_subaray_recv, lower_rank, 201,
        communicator, MPI_STATUS_IGNORE);

    // ������� �� ���������� ���������� ������ �������-������������������ �������

    // ���������� ������ � �������������, ��� ������������ �� ����������� x ����������� ������� �� 2 ��������,
    // �.� ������� ���������� ���� �� �� ��� ��� ������� � ������ �����������.
    // ��� ��, ������ ���������� �� ����������� x ������, ��� ������ PML - ������� (������ PML - 10-15 �����)
    if (rank_coords[0] == 0)
    {
        for (long int j = 2; j < _size_j - 2; ++j)
        {
            for (long int k = 2; k < _size_k - 2; ++k)
            {
                // ������ ������ PML 
                for (long int i = 2; i < 2 + PML_Size; ++i)
                {
                    // ������������ ������� d(s)
                    double demp = static_cast<double>(PML_Size - (i - 2)) / static_cast<double>(PML_Size) * sigma_max;

                    // ������� 9-�� ����������� ��������� �������� � PML �������
                    //this->_PML_transfer_eq_x(i * _I + j * _J + k, demp);
                    index = i * _I + j * _J + k;
                    this->_PML_solve_system_eq(index, g_rank_curr[index - 2 * _I], g_rank_curr[index - _I], g_rank_curr[index], g_rank_curr[index + _I], g_rank_curr[index + 2 * _I], demp);
                }

                // ������ ������ �������� �������������� �������
                for (long int i = 2 + PML_Size; i < _size_i - 2; ++i)
                {
                    // ������� 9-�� ����������� ��������� ��������
                    //this->_transfer_eq_x(i * _I + j * _J + k);
                    index = i * _I + j * _J + k;
                    this->_solve_system_eq(index, g_rank_curr[index - 2 * _I], g_rank_curr[index - _I], g_rank_curr[index], g_rank_curr[index + _I], g_rank_curr[index + 2 * _I]);
                }
            }
        }
    }

    // PML ������� �������������� � ���������, ��� ���������� ��������� � ������� �� ��� x.
    // ��������� �������� ���������� ������� ������ �������� �������
    if ((rank_coords[0] != 0) && (rank_coords[0] != dims[0] - 1))
    {
        for (long int j = 2; j < _size_j - 2; ++j)
        {
            for (long int k = 2; k < _size_k - 2; ++k)
            {
                // ������ ������ �������� �������������� �������
                for (long int i = 2; i < _size_i - 2; ++i)
                {
                    // ������� 9-�� ����������� ��������� ��������
                    //this->_transfer_eq_x(i * _I + j * _J + k);
                    index = i * _I + j * _J + k;
                    this->_solve_system_eq(index, g_rank_curr[index - 2 * _I], g_rank_curr[index - _I], g_rank_curr[index], g_rank_curr[index + _I], g_rank_curr[index + 2 * _I]);
                }
            }
        }
    }

    // ���������� ������� ��������� � �������, � ��� �� �������� PML - �������
    if (rank_coords[0] == dims[0] - 1)
    {
        for (long int j = 2; j < _size_j - 2; ++j)
        {
            for (long int k = 2; k < _size_k - 2; ++k)
            {
                // ������ ������ �������� �������������� �������
                for (long int i = 2; i < _size_i - PML_Size - 2; ++i)
                {
                    // ������� 9-�� ����������� ��������� ��������
                    //this->_transfer_eq_x(i * _I + j * _J + k);
                    index = i * _I + j * _J + k;
                    this->_solve_system_eq(index, g_rank_curr[index - 2 * _I], g_rank_curr[index - _I], g_rank_curr[index], g_rank_curr[index + _I], g_rank_curr[index + 2 * _I]);
                }

                // ������ ������ PML
                for (long int i = _size_i - PML_Size - 2; i < _size_i - 2; ++i)
                {
                    // ������������ ������� d(s)
                    double demp = static_cast<double>((i + 2) - _size_i + PML_Size + 1) / static_cast<double>(PML_Size) * sigma_max;

                    // ������� 9-�� ����������� ��������� �������� � PML �������
                    //this->_PML_transfer_eq_x(i * _I + j * _J + k, demp);
                    index = i * _I + j * _J + k;
                    this->_PML_solve_system_eq(index, g_rank_curr[index - 2 * _I], g_rank_curr[index - _I], g_rank_curr[index], g_rank_curr[index + _I], g_rank_curr[index + 2 * _I], demp);
                }
            }
        }
    }

    // �������� ������ - ������� �� ������������������ ������� � ��������
    for (int i = 2; i < _size_i - 2; ++i)
    {
        for (int j = 2; j < _size_j - 2; ++j)
        {
            for (int k = 2; k < _size_k - 2; ++k)
            {
                index = i * _I + j * _J + k;
                _back_change_1(index);
                w_rank_curr[index] = w_rank_next[index];
            }
        }
    }

    Vars w_0, w_m, w_p1, w_p2;
    // ����������� ��������� ������� �� ����������� x. ������ � ������� ������ ����� �� ������ ������� PML �������.
    // ��� ������� (i = 2)
    // ��� ������ ���������� (rank_coords[0] == 0), "������" ������ ���,
    // � ������� ������, ��� ������ ���� ������ ��������� ���� �� ������,
    // ������������ ������ ��������� �������
    if (rank_coords[0] == 0)
    {
        for (long int j = 2; j < _size_j - 2; ++j)
        {
            for (long int k = 2; k < _size_k - 2; ++k)
            {
                // i = 2
                index = 2 * _I + j * _J + k;

                // ��� ������ ������ �����
                w_p1 = w_rank_curr[index];
                w_0 = w_rank_curr[index - _I];
                w_m = w_rank_curr[index - 2 * _I];
                w_p2 = w_rank_curr[index + _I];

                w_0.u_x = w_p1.u_x;
                w_0.u_y = w_p1.u_y;
                w_0.u_z = w_p1.u_z;

                w_0.sigma_xx = -w_p1.sigma_xx;
                w_0.sigma_xy = -w_p1.sigma_xy;
                w_0.sigma_zx = -w_p1.sigma_zx;

                w_0.sigma_yy = 0.;
                w_0.sigma_yz = 0.;
                w_0.sigma_zz = 0.;

                // ��� ������ ������ �����
                w_m.u_x = w_p2.u_x;
                w_m.u_y = w_p2.u_y;
                w_m.u_z = w_p2.u_z;

                w_m.sigma_xx = -w_p2.sigma_xx;
                w_m.sigma_xy = -w_p2.sigma_xy;
                w_m.sigma_zx = -w_p2.sigma_zx;

                w_m.sigma_yy = 0.;
                w_m.sigma_yz = 0.;
                w_m.sigma_zz = 0.;

                w_rank_curr[index - _I] = w_0;
                w_rank_curr[index - 2 * _I] = w_m;
            }
        }
    }


    // ����������� ��������� ������� �� ����������� x. ������ � ������� ������ ����� �� ������ ������� PML �������.
    // ��� ������� (i = _size_i - 3)
    // ��� ������ ���������� (rank_coords[0] == dims[0] - 1), "������" ������ ���,
    // � ������� ������, ��� ������ ���� ������ ��������� ���� �� ������,
    // ������������ ������ ��������� �������
    if (rank_coords[0] == dims[0] - 1)
    {
        for (long int j = 2; j < _size_j - 2; ++j)
        {
            for (long int k = 2; k < _size_k - 2; ++k)
            {
                // i = _size_i - 3
                index = (_size_i - 3) * _I + j * _J + k;

                // ��� ������ ������ �����
                w_p1 = w_rank_curr[index];
                w_0 = w_rank_curr[index + _I];
                w_m = w_rank_curr[index + 2 * _I];
                w_p2 = w_rank_curr[index - _I];

                w_0.u_x = w_p1.u_x;
                w_0.u_y = w_p1.u_y;
                w_0.u_z = w_p1.u_z;

                w_0.sigma_xx = -w_p1.sigma_xx;
                w_0.sigma_xy = -w_p1.sigma_xy;
                w_0.sigma_zx = -w_p1.sigma_zx;

                w_0.sigma_yy = 0.;
                w_0.sigma_yz = 0.;
                w_0.sigma_zz = 0.;

                // ��� ������ ������ �����
                w_m.u_x = w_p2.u_x;
                w_m.u_y = w_p2.u_y;
                w_m.u_z = w_p2.u_z;

                w_m.sigma_xx = -w_p2.sigma_xx;
                w_m.sigma_xy = -w_p2.sigma_xy;
                w_m.sigma_zx = -w_p2.sigma_zx;

                w_m.sigma_yy = 0.;
                w_m.sigma_yz = 0.;
                w_m.sigma_zz = 0.;

                w_rank_curr[index + _I] = w_0;
                w_rank_curr[index + 2 * _I] = w_m;
            }
        }
    }
}

void Wave3d::_make_step_Y()
{
    // ������� ���������
    // dW/dt + A * dW/dy = 0 

    long int index = 0;
    // ������� � ������������������ �������
    // (������� ������� � ����������� ������)
    for (int i = 2; i < _size_i - 2; ++i)
    {
        for (int j = 0; j < _size_j; ++j)
        {
            for (int k = 2; k < _size_k - 2; ++k)
            {
                index = i * _I + j * _J + k;
                _change_2(index);
            }
        }
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //      rank_coords[1] = 0                rank_coords[1] = 1                    rank_coords[1] = dims[1]-2       rank_coords[1] = dims[1]-1
    // +----+----------------+----+      +----+---------------+----+               +----+---------------+----+      +----+----------------+----+
    // | xx |                | xx |      | xx |               | xx |               | xx |               | xx |      | xx |                | xx |
    // +----+-----+-----+----+----+      +----+----+-----+----+----+               +----+----+-----+----+----+      +----+----+-----+-----+----+
    // | BC | PML |     | ** | ++ |  ->  | ** | ++ |     | ** | ++ |  ->  ...  ->  | ** | ++ |     | ** | ++ |  ->  | ** | ++ |     | PML | BC |
    // | BC | PML |     | ** | ++ |      | ** | ++ |     | ** | ++ |      ...      | ** | ++ |     | ** | ++ |      | ** | ++ |     | PML | BC |
    // | BC | PML |     | ** | ++ |      | ** | ++ |     | ** | ++ |      ...      | ** | ++ |     | ** | ++ |      | ** | ++ |     | PML | BC |
    // | BC | PML |     | ** | ++ |  <-  | ** | ++ |     | ** | ++ |  <-  ...  <-  | ** | ++ |     | ** | ++ |  <-  | ** | ++ |     | PML | BC |
    // +----+-----+-----+----+----+      +----+----+-----+----+----+               +----+----+-----+----+----+      +----+----+-----+-----+----+
    // | xx |                | xx |      | xx |               | xx |               | xx |               | xx |      | xx |                | xx |
    // +----+----------------+----+      +----+---------------+----+               +----+---------------+----+      +----+----------------+----+
    //
    // -> ����� ������������� ������ ����� ���������� ��� ������� ��������� dW/dt + A * dW/dy = 0
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // ������ ������ ����� ��������
    // ���������� ������ "������" ������, ��������� ������ �� "������" �����.
    // � ������, ���� "������" ������ ���, �� right_rank = MPI_PROC_NULL,
    // � ���� ������ ����� ������� MPI_Sendrecv ���������� ��� ������, �������� �������� �� ������������,
    // ����� ���������� ������ ����������.
    // ����������, ���� "�����" ����� �����������, ���������� ������ �������� ������ ��������� ��������.
    MPI_Sendrecv(
        g_rank_curr, 1, right_subaray_send, right_rank, 300,
        g_rank_curr, 1, left_subaray_recv, left_rank, 300,
        communicator, MPI_STATUS_IGNORE);

    // ���������� ����������� ������ ������� MPI_Sendrecv
    MPI_Sendrecv(
        g_rank_curr, 1, left_subaray_send, left_rank, 301,
        g_rank_curr, 1, right_subaray_recv, right_rank, 301,
        communicator, MPI_STATUS_IGNORE);

    // ������� �� ���������� ���������� ������ �������-������������������ �������

    // ���������� ������ � �������������, ��� ������������ �� ����������� y ����������� ������� �� 2 ��������,
    // �.� ������� ���������� ���� �� �� ��� ��� ������� � ������ �����������.
    // ��� ��, ������ ���������� �� ����������� y ������, ��� ������ PML - ������� (������ PML - 10-15 �����)

    if (rank_coords[1] == 0)
    {
        for (long int i = 2; i < _size_i - 2; ++i)
        {
            for (long int k = 2; k < _size_k - 2; ++k)
            {
                // ������ ������ PML 
                for (long int j = 2; j < 2 + PML_Size; ++j)
                {
                    // ������������ ������� d(s)
                    double demp = static_cast<double>(PML_Size - (j - 2)) / static_cast<double>(PML_Size) * sigma_max;

                    // ������� 9-�� ����������� ��������� �������� � PML �������
                    //this->_PML_transfer_eq_y(i * _I + j * _J + k, demp);
                    index = i * _I + j * _J + k;
                    this->_PML_solve_system_eq(index, g_rank_curr[index - 2 * _J], g_rank_curr[index - _J], g_rank_curr[index], g_rank_curr[index + _J], g_rank_curr[index + 2 * _J], demp);
                }

                // ������ ������ �������� �������������� �������
                for (long int j = 2 + PML_Size; j < _size_j - 2; ++j)
                {
                    // ������� 9-�� ����������� ��������� ��������
                    //this->_transfer_eq_y(i * _I + j * _J + k);
                    index = i * _I + j * _J + k;
                    this->_solve_system_eq(index, g_rank_curr[index - 2 * _J], g_rank_curr[index - _J], g_rank_curr[index], g_rank_curr[index + _J], g_rank_curr[index + 2 * _J]);
                }
            }
        }
    }

    // PML ������� �������������� � ���������, ��� ���������� ��������� � ������� �� ��� j.
    // ��������� �������� ���������� ������� ������ �������� �������
    if ((rank_coords[1] != 0) && (rank_coords[1] != dims[1] - 1))
    {
        for (long int i = 2; i < _size_i - 2; ++i)
        {
            for (long int k = 2; k < _size_k - 2; ++k)
            {
                // ������ ������ �������� �������������� �������
                // ������ j - ���������, �.�. ���������� �� ������ �� ����� ����������;
                for (long int j = 2; j < _size_j - 2; ++j)
                {
                    // ������� 9-�� ����������� ��������� ��������
                    //this->_transfer_eq_y(i * _I + j * _J + k);
                    index = i * _I + j * _J + k;
                    this->_solve_system_eq(index, g_rank_curr[index - 2 * _J], g_rank_curr[index - _J], g_rank_curr[index], g_rank_curr[index + _J], g_rank_curr[index + 2 * _J]);
                }
            }
        }
    }

    // ���������� ������� ��������� � �������, � ��� �� �������� PML - �������
    if (rank_coords[1] == dims[1] - 1)
    {
        for (long int i = 2; i < _size_i - 2; ++i)
        {
            for (long int k = 2; k < _size_k - 2; ++k)
            {
                // ������ ������ �������� �������������� �������
                for (long int j = 2; j < _size_j - PML_Size - 2; ++j)
                {
                    // ������� 9-�� ����������� ��������� ��������
                    //this->_transfer_eq_y(i * _I + j * _J + k);
                    index = i * _I + j * _J + k;
                    this->_solve_system_eq(index, g_rank_curr[index - 2 * _J], g_rank_curr[index - _J], g_rank_curr[index], g_rank_curr[index + _J], g_rank_curr[index + 2 * _J]);
                }

                // ������ ������ PML
                for (long int j = _size_j - PML_Size - 2; j < _size_j - 2; ++j)
                {
                    // ������������ ������� d(s)
                    double demp = static_cast<double>((j + 2) - _size_j + PML_Size + 1) / static_cast<double>(PML_Size) * sigma_max;

                    // ������� 9-�� ����������� ��������� �������� � PML �������
                    //this->_PML_transfer_eq_y(i * _I + j * _J + k, demp);
                    index = i * _I + j * _J + k;
                    this->_PML_solve_system_eq(index, g_rank_curr[index - 2 * _J], g_rank_curr[index - _J], g_rank_curr[index], g_rank_curr[index + _J], g_rank_curr[index + 2 * _J], demp);
                }
            }
        }
    }


    // �������� ������ - ������� �� ������������������ ������� � ��������
    for (int i = 2; i < _size_i - 2; ++i)
    {
        for (int j = 2; j < _size_j - 2; ++j)
        {
            for (int k = 2; k < _size_k - 2; ++k)
            {
                index = i * _I + j * _J + k;
                _back_change_2(index);
                w_rank_curr[index] = w_rank_next[index];
            }
        }
    }

    Vars w_0, w_m, w_p1, w_p2;
    // ����������� ��������� ������� �� ����������� y. ������ � ������� ������ ����� �� ������ ������� PML �������.
    // ��� ������� ����� (j = 2)
    // j = 0, 1 - �������������� ������ �����
    //
    // ��� ������ ���������� (rank_coords[1] == 0), "������" ����� ���,
    // � ������� ������, ��� ������ ���� ������ ��������� ���� �� ������,
    // ������������ ������ ��������� �������
    if (rank_coords[1] == 0)
    {
        for (long int i = 2; i < _size_i - 2; ++i)
        {
            for (long int k = 2; k < _size_k - 2; ++k)
            {
                // j = 2
                index = i * _I + 2 * _J + k;

                // ��� ������ ������ �����
                w_p1 = w_rank_curr[index];
                w_0 = w_rank_curr[index - _J];
                w_m = w_rank_curr[index - 2 * _J];
                w_p2 = w_rank_curr[index + _J];

                w_0.u_x = w_p1.u_x;
                w_0.u_y = w_p1.u_y;
                w_0.u_z = w_p1.u_z;

                w_0.sigma_xy = -w_p1.sigma_xy;
                w_0.sigma_yy = -w_p1.sigma_yy;
                w_0.sigma_yz = -w_p1.sigma_yz;

                w_0.sigma_xx = 0.;
                w_0.sigma_zx = 0.;
                w_0.sigma_zz = 0.;

                // ��� ������ ������ �����
                w_m.u_x = w_p2.u_x;
                w_m.u_y = w_p2.u_y;
                w_m.u_z = w_p2.u_z;

                w_m.sigma_xy = -w_p2.sigma_xy;
                w_m.sigma_yy = -w_p2.sigma_yy;
                w_m.sigma_yz = -w_p2.sigma_yz;

                w_m.sigma_xx = 0.;
                w_m.sigma_zx = 0.;
                w_m.sigma_zz = 0.;

                w_rank_curr[index - 2 * _J] = w_m;
                w_rank_curr[index - _J] = w_0;
            }
        }
    }


    // ����������� ��������� ������� �� ����������� y. ������ � ������� ������ ����� �� ������ ������� PML �������.
    // ��� ������� ����� (j = _size_j - 3)
    // j = _size_j - 2, _size_j - 1 - �������������� ������ �����
    //
    // ��� ������ ���������� (rank_coords[1] == dims[1] - 1), "������" ������ ���,
    // � ������� ������, ��� ������ ���� ������ ��������� ���� �� ������,
    // ������������ ������ ��������� �������
    if (rank_coords[1] == dims[1] - 1)
    {
        for (long int i = 2; i < _size_i - 2; ++i)
        {
            for (long int k = 2; k < _size_k - 2; ++k)
            {
                // j = _size_j - 3
                index = i * _I + (_size_j - 3) * _J + k;

                // ��� ������ ������ �����
                w_p1 = w_rank_curr[index];
                w_0 = w_rank_curr[index + _J];
                w_m = w_rank_curr[index + 2 * _J];
                w_p2 = w_rank_curr[index - _J];

                w_0.u_x = w_p1.u_x;
                w_0.u_y = w_p1.u_y;
                w_0.u_z = w_p1.u_z;

                w_0.sigma_xy = -w_p1.sigma_xy;
                w_0.sigma_yy = -w_p1.sigma_yy;
                w_0.sigma_yz = -w_p1.sigma_yz;

                w_0.sigma_xx = 0.;
                w_0.sigma_zx = 0.;
                w_0.sigma_zz = 0.;

                // ��� ������ ������ �����
                w_m.u_x = w_p2.u_x;
                w_m.u_y = w_p2.u_y;
                w_m.u_z = w_p2.u_z;

                w_m.sigma_xy = -w_p2.sigma_xy;
                w_m.sigma_yy = -w_p2.sigma_yy;
                w_m.sigma_yz = -w_p2.sigma_yz;

                w_m.sigma_xx = 0.;
                w_m.sigma_zx = 0.;
                w_m.sigma_zz = 0.;

                w_rank_curr[index + 2 * _J] = w_m;
                w_rank_curr[index + _J] = w_0;
            }
        }
    }
}

void Wave3d::_make_step_Z()
{
    // ������� ���������
    // dW/dt + A * dW/dz = 0 

    long int index = 0;
    // ������� � ������������������ �������
    for (int i = 2; i < _size_i - 2; ++i)
    {
        for (int j = 2; j < _size_j - 2; ++j)
        {
            for (int k = 0; k < _size_k; ++k)
            {
                index = i * _I + j * _J + k;
                _change_3(index);
            }
        }
    }


    // ������� �� ���������� ���������� ������ �������-������������������ �������
    for (long int i = 2; i < _size_i - 2; ++i)
    {
        for (long int j = 2; j < _size_j - 2; ++j)
        {
            // ������ ������ PML
            for (long int k = 2; k < 2 + PML_Size; ++k)
            {
                // ������������ ������� d(s)
                double demp = static_cast<double>(PML_Size - (k - 2)) / static_cast<double>(PML_Size) * sigma_max;

                // ������� 9-�� ����������� ��������� �������� � PML �������
                //this->_PML_transfer_eq_z(i * _I + j * _J + k, demp);
                index = i * _I + j * _J + k;
                this->_PML_solve_system_eq(index, g_rank_curr[index - 2], g_rank_curr[index - 1], g_rank_curr[index], g_rank_curr[index + 1], g_rank_curr[index + 2], demp);
            }


            // ������ ������ �������� �������������� �������
            for (long int k = 2 + PML_Size; k < 2 + I + PML_Size; ++k)
            {
                // ������� 9-�� ����������� ��������� ��������
                //this->_transfer_eq_z(i * _I + j * _J + k);
                index = i * _I + j * _J + k;
                this->_solve_system_eq(index, g_rank_curr[index - 2], g_rank_curr[index - 1], g_rank_curr[index], g_rank_curr[index + 1], g_rank_curr[index + 2]);
            }


            // ������ ������ PML
            for (long int k = 2 + I + PML_Size; k < 2 + I + 2 * PML_Size; ++k)
            {
                // ������������ ������� d(s)
                double demp = static_cast<double>((k - 2) - I - PML_Size + 1) / static_cast<double>(PML_Size) * sigma_max;

                // ������� 9-�� ����������� ��������� �������� � PML �������
                //this->_PML_transfer_eq_z(i * _I + j * _J + k, demp);
                index = i * _I + j * _J + k;
                this->_PML_solve_system_eq(index, g_rank_curr[index - 2], g_rank_curr[index - 1], g_rank_curr[index], g_rank_curr[index + 1], g_rank_curr[index + 2], demp);
            }
        }
    }


    // �������� ������ - ������� �� ������������������ ������� � ��������
    for (int i = 2; i < _size_i - 2; ++i)
    {
        for (int j = 2; j < _size_j - 2; ++j)
        {
            for (int k = 2; k < _size_k - 2; ++k)
            {
                index = i * _I + j * _J + k;
                _back_change_3(index);
                w_rank_curr[index] = w_rank_next[index];
            }
        }
    }


    Vars w_0, w_m, w_p1, w_p2;
    // ����������� ��������� ������� �� ����������� z. ������ � ������� ������ ����� �� ������ ������� PML �������
    // ��� ������� �����
    // k = 2
    // � = 0, 1 - �������������� ������ �����
    for (long int i = 2; i < _size_i - 2; ++i)
    {
        for (long int j = 2; j < _size_j - 2; ++j)
        {
            // k = 2
            index = i * _I + j * _J + 2;

            // ��� ������ ������ �����
            w_p1 = w_rank_curr[index];
            w_0 = w_rank_curr[index - 1];
            w_m = w_rank_curr[index - 2];
            w_p2 = w_rank_curr[index + 1];

            w_0.u_x = w_p1.u_x;
            w_0.u_y = w_p1.u_y;
            w_0.u_z = w_p1.u_z;

            w_0.sigma_zx = -w_p1.sigma_zx;
            w_0.sigma_yz = -w_p1.sigma_yz;
            w_0.sigma_zz = -w_p1.sigma_zz;

            w_0.sigma_xx = 0.;
            w_0.sigma_xy = 0.;
            w_0.sigma_yy = 0.;

            // ��� ������ ������ �����
            w_m.u_x = w_p2.u_x;
            w_m.u_y = w_p2.u_y;
            w_m.u_z = w_p2.u_z;

            w_m.sigma_zx = -w_p2.sigma_zx;
            w_m.sigma_yz = -w_p2.sigma_yz;
            w_m.sigma_zz = -w_p2.sigma_zz;

            w_m.sigma_xx = 0.;
            w_m.sigma_xy = 0.;
            w_m.sigma_yy = 0.;

            w_rank_curr[index - 2] = w_m;
            w_rank_curr[index - 1] = w_0;
        }
    }


    // ��� ������ �������
    // k = _size_k - 3
    // k = _size_k - 2, _size_k - 1 - �������������� ������ �����
    for (long int i = 2; i < _size_i - 2; ++i)
    {
        for (long int j = 2; j < _size_j - 2; ++j)
        {
            index = i * _I + j * _J + _size_k - 3;

            // ��� ������ ������ �����
            w_p1 = w_rank_curr[index];
            w_0 = w_rank_curr[index + 1];
            w_m = w_rank_curr[index + 2];
            w_p2 = w_rank_curr[index - 1];

            w_0.u_x = w_p1.u_x;
            w_0.u_y = w_p1.u_y;
            w_0.u_z = w_p1.u_z;

            w_0.sigma_zx = -w_p1.sigma_zx;
            w_0.sigma_yz = -w_p1.sigma_yz;
            w_0.sigma_zz = -w_p1.sigma_zz;

            w_0.sigma_xx = 0.;
            w_0.sigma_xy = 0.;
            w_0.sigma_yy = 0.;

            // ��� ������ ������ �����
            w_m.u_x = w_p2.u_x;
            w_m.u_y = w_p2.u_y;
            w_m.u_z = w_p2.u_z;

            w_m.sigma_zx = -w_p2.sigma_zx;
            w_m.sigma_yz = -w_p2.sigma_yz;
            w_m.sigma_zz = -w_p2.sigma_zz;

            w_m.sigma_xx = 0.;
            w_m.sigma_xy = 0.;
            w_m.sigma_yy = 0.;

            w_rank_curr[index + 2] = w_m;
            w_rank_curr[index + 1] = w_0;
        }
    }


    // ��������� ������� - ��������� �� ������� �� ����������� z = 0:
    // sigma_zz = sigma_zy = sigma_zx = 0
    // ��� ���������� u_x, u_y, u_z, sigma_xx, sigma_xy, sigma_yy ��������� ������� �� ������! => PML
    // k = 2 + PML_Size
    for (long int i = 2; i < _size_i - 2; ++i)
    {
        for (long int j = 2; j < _size_j - 2; ++j)
        {
            index = i * _I + j * _J + (2 + PML_Size);
            w_rank_curr[index].sigma_zx = 0.;
            w_rank_curr[index].sigma_yz = 0.;
            w_rank_curr[index].sigma_zz = 0.;
        }
    }
}

void Wave3d::_write_to_file_Sigma()
{
    if (this->current_iter_sigma > this->num_iters_in_file)
    {
        this->current_iter_sigma = 1;
        this->num_current_file_sigma += 1;
        MPI_File_close(&datafile_sigma);
    }

    if ((this->current_iter_sigma > this->N - (num_file - 1) * num_iters_in_file + 1) && (this->num_current_file_sigma == this->num_file))
    {
        MPI_File_close(&datafile_sigma);
    }

    if (this->current_iter_sigma == 1)
    {
        std::string filename_sigma = std::string("Sigma") + std::to_string(this->num_current_file_sigma) + std::string(".bin");
        char* file = new char[filename_sigma.length() + 1];
        strcpy_s(file, filename_sigma.length() + 1, filename_sigma.c_str());

        if (rank == 0)
        {
            MPI_File_delete(file, MPI_INFO_NULL);
        }
        MPI_File_open(communicator, file, MPI_MODE_CREATE | MPI_MODE_EXCL | MPI_MODE_WRONLY, MPI_INFO_NULL, &datafile_sigma);
        MPI_File_set_view(datafile_sigma, 0, MPI_DOUBLE, subarray_rw, "native", MPI_INFO_NULL);

        delete[] file;
    }


    Vars v;
    long int index = 0;
    for (int i = starti_rw; i < endi_rw; i++)
    {
        for (int j = startj_rw; j < endj_rw; j++)
        {
            for (int k = 2 + PML_Size; k < 2 + I + PML_Size; k++)
            {
                v = w_rank_curr[i * _I + j * _J + k];

                buf_rw[index] = (v.sigma_xx + v.sigma_yy + v.sigma_zz) / 3.;
                index++;
            }
        }
    }
    MPI_File_write_all(datafile_sigma, buf_rw, count_rw, MPI_DOUBLE, &status);
    this->current_iter_sigma += 1;
}

void Wave3d::_write_to_file_Sigma_Okt()
{
    if (this->current_iter_sigma_okt > this->num_iters_in_file)
    {
        this->current_iter_sigma_okt = 1;
        this->num_current_file_sigma_okt += 1;
        MPI_File_close(&datafile_sigma_okt);
    }

    if ((this->current_iter_sigma_okt > this->N - (num_file - 1) * num_iters_in_file + 1) && (this->num_current_file_sigma_okt == this->num_file))
    {
        MPI_File_close(&datafile_sigma_okt);
    }

    if (this->current_iter_sigma == 1)
    {
        std::string filename_sigma_okt = std::string("Sigma_okt") + std::to_string(this->num_current_file_sigma_okt) + std::string(".bin");
        char* file = new char[filename_sigma_okt.length() + 1];
        strcpy_s(file, filename_sigma_okt.length() + 1, filename_sigma_okt.c_str());

        if (rank == 0)
        {
            MPI_File_delete(file, MPI_INFO_NULL);
        }
        MPI_File_open(communicator, file, MPI_MODE_CREATE | MPI_MODE_EXCL | MPI_MODE_WRONLY, MPI_INFO_NULL, &datafile_sigma_okt);
        MPI_File_set_view(datafile_sigma_okt, 0, MPI_DOUBLE, subarray_rw, "native", MPI_INFO_NULL);

        delete[] file;
    }

    Vars v;
    long int index = 0;
    for (int i = starti_rw; i < endi_rw; i++)
    {
        for (int j = startj_rw; j < endj_rw; j++)
        {
            for (int k = 2 + PML_Size; k < 2 + I + PML_Size; k++)
            {
                v = w_rank_curr[i * _I + j * _J + k];

                buf_rw[index] = sqrt(
                    (v.sigma_xx - v.sigma_yy) * (v.sigma_xx - v.sigma_yy) +
                    (v.sigma_yy - v.sigma_zz) * (v.sigma_yy - v.sigma_zz) +
                    (v.sigma_zz - v.sigma_xx) * (v.sigma_zz - v.sigma_xx)) / 3.;
                index++;
            }
        }
    }
    MPI_File_write_all(datafile_sigma_okt, buf_rw, count_rw, MPI_DOUBLE, &status);
    this->current_iter_sigma_okt += 1;
}

void Wave3d::_write_to_file_Abs_U()
{
    if (this->current_iter_abs_u > this->num_iters_in_file)
    {
        this->current_iter_abs_u = 1;
        this->num_current_file_abs_u += 1;
        MPI_File_close(&datafile_abs_u);
    }

    if ((this->current_iter_abs_u > this->N - (num_file - 1) * num_iters_in_file + 1) && (this->num_current_file_abs_u == this->num_file))
    {
        MPI_File_close(&datafile_abs_u);
    }

    if (this->current_iter_abs_u == 1)
    {
        std::string filename_abs_u = std::string("Abs_U") + std::to_string(this->num_current_file_abs_u) + std::string(".bin");
        char* file = new char[filename_abs_u.length() + 1];
        strcpy_s(file, filename_abs_u.length() + 1, filename_abs_u.c_str());

        if (rank == 0)
        {
            MPI_File_delete(file, MPI_INFO_NULL);
        }
        MPI_File_open(communicator, file, MPI_MODE_CREATE | MPI_MODE_EXCL | MPI_MODE_WRONLY, MPI_INFO_NULL, &datafile_abs_u);
        MPI_File_set_view(datafile_abs_u, 0, MPI_DOUBLE, subarray_rw, "native", MPI_INFO_NULL);

        delete[] file;
    }

    Vars v;
    long int index = 0;
    for (int i = starti_rw; i < endi_rw; ++i)
    {
        for (int j = startj_rw; j < endj_rw; ++j)
        {
            for (int k = 2 + PML_Size; k < 2 + I + PML_Size; ++k)
            {
                v = w_rank_curr[i * _I + j * _J + k];

                buf_rw[index] = sqrt(v.u_x * v.u_x + v.u_y * v.u_y + v.u_z * v.u_z);
                index++;
            }
        }
    }
    MPI_File_write_all(datafile_abs_u, buf_rw, count_rw, MPI_DOUBLE, &status);
    this->current_iter_abs_u += 1;
}

void Wave3d::_write_to_file_Ux()
{
    if (this->current_iter_u_x > this->num_iters_in_file)
    {
        this->current_iter_u_x = 1;
        this->num_current_file_u_x += 1;
        MPI_File_close(&datafile_u_x);
    }

    if ((this->current_iter_u_x > this->N - (num_file - 1) * num_iters_in_file + 1) && (this->num_current_file_u_x == this->num_file))
    {
        MPI_File_close(&datafile_u_x);
    }

    if (this->current_iter_u_x == 1)
    {
        std::string filename_u_x = std::string("Ux") + std::to_string(this->num_current_file_u_x) + std::string(".bin");
        char* file = new char[filename_u_x.length() + 1];
        strcpy_s(file, filename_u_x.length() + 1, filename_u_x.c_str());

        if (rank == 0)
        {
            MPI_File_delete(file, MPI_INFO_NULL);
        }
        MPI_File_open(communicator, file, MPI_MODE_CREATE | MPI_MODE_EXCL | MPI_MODE_WRONLY, MPI_INFO_NULL, &datafile_u_x);
        MPI_File_set_view(datafile_u_x, 0, MPI_DOUBLE, subarray_rw, "native", MPI_INFO_NULL);

        delete[] file;
    }

    long int index = 0;
    for (int i = starti_rw; i < endi_rw; i++)
    {
        for (int j = startj_rw; j < endj_rw; j++)
        {
            for (int k = 2 + PML_Size; k < 2 + I + PML_Size; k++)
            {
                buf_rw[index] = w_rank_curr[i * _I + j * _J + k].u_x;
                index++;
            }
        }
    }
    MPI_File_write_all(datafile_u_x, buf_rw, count_rw, MPI_DOUBLE, &status);
    this->current_iter_u_x += 1;
}

void Wave3d::_write_to_file_Uy()
{
    if (this->current_iter_u_y > this->num_iters_in_file)
    {
        this->current_iter_u_y = 1;
        this->num_current_file_u_y += 1;
        MPI_File_close(&datafile_u_y);
    }

    if ((this->current_iter_u_y > this->N - (num_file - 1) * num_iters_in_file + 1) && (this->num_current_file_u_y == this->num_file))
    {
        MPI_File_close(&datafile_u_y);
    }

    if (this->current_iter_u_y == 1)
    {
        std::string filename_u_y = std::string("Uy") + std::to_string(this->num_current_file_u_y) + std::string(".bin");
        char* file = new char[filename_u_y.length() + 1];
        strcpy_s(file, filename_u_y.length() + 1, filename_u_y.c_str());

        if (rank == 0)
        {
            MPI_File_delete(file, MPI_INFO_NULL);
        }
        MPI_File_open(communicator, file, MPI_MODE_CREATE | MPI_MODE_EXCL | MPI_MODE_WRONLY, MPI_INFO_NULL, &datafile_u_y);
        MPI_File_set_view(datafile_u_y, 0, MPI_DOUBLE, subarray_rw, "native", MPI_INFO_NULL);

        delete[] file;
    }

    long int index = 0;
    for (int i = starti_rw; i < endi_rw; i++)
    {
        for (int j = startj_rw; j < endj_rw; j++)
        {
            for (int k = 2 + PML_Size; k < 2 + I + PML_Size; k++)
            {
                buf_rw[index] = w_rank_curr[i * _I + j * _J + k].u_y;
                index++;
            }
        }
    }
    MPI_File_write_all(datafile_u_y, buf_rw, count_rw, MPI_DOUBLE, &status);
    this->current_iter_u_y += 1;
}

void Wave3d::_write_to_file_Uz()
{
    if (this->current_iter_u_z > this->num_iters_in_file)
    {
        this->current_iter_u_z = 1;
        this->num_current_file_u_z += 1;
        MPI_File_close(&datafile_u_z);
    }

    if ((this->current_iter_u_z > this->N - (num_file - 1) * num_iters_in_file + 1) && (this->num_current_file_u_z == this->num_file))
    {
        MPI_File_close(&datafile_u_z);
    }

    if (this->current_iter_u_z == 1)
    {
        std::string filename_u_z = std::string("Uz") + std::to_string(this->num_current_file_u_z) + std::string(".bin");
        char* file = new char[filename_u_z.length() + 1];
        strcpy_s(file, filename_u_z.length() + 1, filename_u_z.c_str());

        if (rank == 0)
        {
            MPI_File_delete(file, MPI_INFO_NULL);
        }
        MPI_File_open(communicator, file, MPI_MODE_CREATE | MPI_MODE_EXCL | MPI_MODE_WRONLY, MPI_INFO_NULL, &datafile_u_z);
        MPI_File_set_view(datafile_u_z, 0, MPI_DOUBLE, subarray_rw, "native", MPI_INFO_NULL);

        delete[] file;
    }

    long int index = 0;
    for (int i = starti_rw; i < endi_rw; i++)
    {
        for (int j = startj_rw; j < endj_rw; j++)
        {
            for (int k = 2 + PML_Size; k < 2 + I + PML_Size; k++)
            {
                buf_rw[index] = w_rank_curr[i * _I + j * _J + k].u_z;
                index++;
            }
        }
    }
    MPI_File_write_all(datafile_u_z, buf_rw, count_rw, MPI_DOUBLE, &status);
    this->current_iter_u_z += 1;
}

void Wave3d::_read_param_from_file()
{
    // ������ �� ����� ��������� ����� vp - �������� ���������� ����
    MPI_File datafile_vp;
    std::string filename_vp = std::string("Vp.bin");
    char* file_vp = new char[filename_vp.length() + 1];
    strcpy_s(file_vp, filename_vp.length() + 1, filename_vp.c_str());

    MPI_File_open(communicator, file_vp, MPI_MODE_RDONLY, MPI_INFO_NULL, &datafile_vp);
    MPI_File_set_view(datafile_vp, 0, MPI_DOUBLE, subarray_rw, "native", MPI_INFO_NULL);
    MPI_File_read_all(datafile_vp, buf_rw, count_rw, MPI_DOUBLE, &status);

    //int count;
    //MPI_Get_count(&status, MPI_DOUBLE, &count);
    //printf("process (%d, %d) read %d\n", rank_coords[0], rank_coords[1], count);

    long int index = 0;
    for (int i = starti_rw; i < endi_rw; ++i)
    {
        for (int j = startj_rw; j < endj_rw; ++j)
        {
            for (int k = 2 + PML_Size; k < I + 2 + PML_Size; ++k)
            {
                v_p[i * _I + j * _J + k] = buf_rw[index];
                index++;
            }
        }
    }
    MPI_File_close(&datafile_vp);
    delete[] file_vp;

    // ������ �� ����� ��������� ����� vs - �������� ���������� ����
    MPI_File datafile_vs;
    std::string filename_vs = std::string("Vs.bin");
    char* file_vs = new char[filename_vs.length() + 1];
    strcpy_s(file_vs, filename_vs.length() + 1, filename_vs.c_str());

    MPI_File_open(communicator, file_vs, MPI_MODE_RDONLY, MPI_INFO_NULL, &datafile_vs);
    MPI_File_set_view(datafile_vs, 0, MPI_DOUBLE, subarray_rw, "native", MPI_INFO_NULL);
    MPI_File_read_all(datafile_vs, buf_rw, count_rw, MPI_DOUBLE, &status);

    index = 0;
    for (int i = starti_rw; i < endi_rw; i++)
    {
        for (int j = startj_rw; j < endj_rw; j++)
        {
            for (int k = 2 + PML_Size; k < I + 2 + PML_Size; k++)
            {
                v_s[i * _I + j * _J + k] = buf_rw[index];
                index++;
            }
        }
    }
    MPI_File_close(&datafile_vs);
    delete[] file_vs;

    // ������ �� ����� ��������� ����� rho - ���������
    MPI_File datafile_rho;
    std::string filename_rho = std::string("Rho.bin");
    char* file_rho = new char[filename_rho.length() + 1];
    strcpy_s(file_rho, filename_rho.length() + 1, filename_rho.c_str());

    MPI_File_open(communicator, file_rho, MPI_MODE_RDONLY, MPI_INFO_NULL, &datafile_rho);
    MPI_File_set_view(datafile_rho, 0, MPI_DOUBLE, subarray_rw, "native", MPI_INFO_NULL);
    MPI_File_read_all(datafile_rho, buf_rw, count_rw, MPI_DOUBLE, &status);

    index = 0;
    for (int i = starti_rw; i < endi_rw; i++)
    {
        for (int j = startj_rw; j < endj_rw; j++)
        {
            for (int k = 2 + PML_Size; k < I + 2 + PML_Size; k++)
            {
                Rho[i * _I + j * _J + k] = buf_rw[index];
                index++;
            }
        }
    }
    MPI_File_close(&datafile_rho);
    delete[] file_rho;

    // ��������� ����� ������� �� ����� ��� �������� �������������� ������� ��� PML.
    // �������������� ��� ������ �� PML - ������� � ��������� �����.

    long int index_1;
    long int index_2;
    
    double _vp;
    double _vs;
    double rho;

    /////////////////////////////////////////////////////////////
    // 
    // +++++++++
    // ++ ******
    // ++ ******
    // ++ ******
    //
    /////////////////////////////////////////////////////////////
    if ((rank_coords[0] == 0) && (rank_coords[1] == 0))
    {
        for (int i = 0; i < 2 + PML_Size; ++i)
        {
            for (int j = 0; j < 2 + PML_Size; ++j)
            {
                index_1 = i * _I + j * _J;
                index_2 = (2 + PML_Size) * _I + (2 + PML_Size) * _J + (2 + PML_Size);

                _vp = v_p[index_2];
                _vs = v_s[index_2];
                rho = Rho[index_2];

                for (int k = 0; k < 2 + PML_Size; ++k)
                {
                    v_p[index_1] = _vp;
                    v_s[index_1] = _vs;
                    Rho[index_1] = rho;
                    ++index_1;
                }

                for (int k = 2 + PML_Size; k < I + 2 + PML_Size; ++k)
                {
                    v_p[index_1] = v_p[index_2];
                    v_s[index_1] = v_s[index_2];
                    Rho[index_1] = Rho[index_2];
                    ++index_1;
                    ++index_2;
                }

                --index_2;
                _vp = v_p[index_2];
                _vs = v_s[index_2];
                rho = Rho[index_2];

                for (int k = I + 2 + PML_Size; k < 4 + I + 2 * PML_Size; ++k)
                {
                    v_p[index_1] = _vp;
                    v_s[index_1] = _vs;
                    Rho[index_1] = rho;
                    ++index_1;
                }
            }

            for (int j = 2 + PML_Size; j < _size_j - 2; ++j)
            {
                index_1 = i * _I + j * _J;
                index_2 = (2 + PML_Size) * _I + j * _J + (2 + PML_Size);

                _vp = v_p[index_2];
                _vs = v_s[index_2];
                rho = Rho[index_2];

                for (int k = 0; k < 2 + PML_Size; ++k)
                {
                    v_p[index_1] = _vp;
                    v_s[index_1] = _vs;
                    Rho[index_1] = rho;
                    ++index_1;
                }

                for (int k = 2 + PML_Size; k < I + 2 + PML_Size; ++k)
                {
                    v_p[index_1] = v_p[index_2];
                    v_s[index_1] = v_s[index_2];
                    Rho[index_1] = Rho[index_2];
                    ++index_1;
                    ++index_2;
                }

                --index_2;
                _vp = v_p[index_2];
                _vs = v_s[index_2];
                rho = Rho[index_2];

                for (int k = I + 2 + PML_Size; k < 4 + I + 2 * PML_Size; ++k)
                {
                    v_p[index_1] = _vp;
                    v_s[index_1] = _vs;
                    Rho[index_1] = rho;
                    ++index_1;
                }
            }
        }

        for (int i = 2 + PML_Size; i < _size_i - 2; ++i)
        {
            for (int j = 0; j < 2 + PML_Size; ++j)
            {
                index_1 = i * _I + j * _J;
                index_2 = i * _I + (2 + PML_Size) * _J + (2 + PML_Size);

                _vp = v_p[index_2];
                _vs = v_s[index_2];
                rho = Rho[index_2];

                for (int k = 0; k < 2 + PML_Size; ++k)
                {
                    v_p[index_1] = _vp;
                    v_s[index_1] = _vs;
                    Rho[index_1] = rho;
                    ++index_1;
                }

                for (int k = 2 + PML_Size; k < I + 2 + PML_Size; ++k)
                {
                    v_p[index_1] = v_p[index_2];
                    v_s[index_1] = v_s[index_2];
                    Rho[index_1] = Rho[index_2];
                    ++index_1;
                    ++index_2;
                }

                --index_2;
                _vp = v_p[index_2];
                _vs = v_s[index_2];
                rho = Rho[index_2];

                for (int k = I + 2 + PML_Size; k < 4 + I + 2 * PML_Size; ++k)
                {
                    v_p[index_1] = _vp;
                    v_s[index_1] = _vs;
                    Rho[index_1] = rho;
                    ++index_1;
                }
            }

            for (int j = 2 + PML_Size; j < _size_j - 2; ++j)
            {
                index_1 = i * _I + j * _J;
                index_2 = i * _I + j * _J + 2 + PML_Size;

                _vp = v_p[index_2];
                _vs = v_s[index_2];
                rho = Rho[index_2];

                for (int k = 0; k < 2 + PML_Size; ++k)
                {
                    v_p[index_1] = _vp;
                    v_s[index_1] = _vs;
                    Rho[index_1] = rho;
                    ++index_1;
                }

                index_1 += I;
                index_2 += I - 1;
                _vp = v_p[index_2];
                _vs = v_s[index_2];
                rho = Rho[index_2];

                for (int k = 2 + I + PML_Size; k < 4 + I + 2 * PML_Size; ++k)
                {
                    v_p[index_1] = _vp;
                    v_s[index_1] = _vs;
                    Rho[index_1] = rho;
                    ++index_1;
                }
            }
        }
    }

    /////////////////////////////////////////////////////////////
    // 
    // ++ ******
    // ++ ******
    // ++ ******
    // +++++++++
    //
    /////////////////////////////////////////////////////////////
    if ((rank_coords[0] == dims[0] - 1) && (rank_coords[1] == 0))
    {
        for (int i = 2; i < _size_i - PML_Size - 2; ++i)
        {
            for (int j = 0; j < 2 + PML_Size; ++j)
            {
                index_1 = i * _I + j * _J;
                index_2 = i * _I + (2 + PML_Size) * _J + 2 + PML_Size;

                _vp = v_p[index_2];
                _vs = v_s[index_2];
                rho = Rho[index_2];

                for (int k = 0; k < 2 + PML_Size; ++k)
                {
                    v_p[index_1] = _vp;
                    v_s[index_1] = _vs;
                    Rho[index_1] = rho;
                    ++index_1;
                }

                for (int k = 2 + PML_Size; k < I + 2 + PML_Size; ++k)
                {
                    v_p[index_1] = v_p[index_2];
                    v_s[index_1] = v_s[index_2];
                    Rho[index_1] = Rho[index_2];
                    ++index_1;
                    ++index_2;
                }

                --index_2;
                _vp = v_p[index_2];
                _vs = v_s[index_2];
                rho = Rho[index_2];

                for (int k = I + 2 + PML_Size; k < 4 + I + 2 * PML_Size; ++k)
                {
                    v_p[index_1] = _vp;
                    v_s[index_1] = _vs;
                    Rho[index_1] = rho;
                    ++index_1;
                }
            }

            for (int j = 2 + PML_Size; j < _size_j - 2; ++j)
            {
                index_1 = i * _I + j * _J;
                index_2 = i * _I + j * _J + 2 + PML_Size;

                _vp = v_p[index_2];
                _vs = v_s[index_2];
                rho = Rho[index_2];

                for (int k = 0; k < 2 + PML_Size; ++k)
                {
                    v_p[index_1] = _vp;
                    v_s[index_1] = _vs;
                    Rho[index_1] = rho;
                    ++index_1;
                }

                index_1 += I;
                index_2 += I - 1;

                _vp = v_p[index_2];
                _vs = v_s[index_2];
                rho = Rho[index_2];

                for (int k = I + 2 + PML_Size; k < 4 + I + 2 * PML_Size; ++k)
                {
                    v_p[index_1] = _vp;
                    v_s[index_1] = _vs;
                    Rho[index_1] = rho;
                    ++index_1;
                }
            }
        }

        for (int i = _size_i - PML_Size - 2; i < _size_i; ++i)
        {
            for (int j = 0; j < 2 + PML_Size; ++j)
            {
                index_1 = i * _I + j * _J;
                index_2 = (_size_i - 3 - PML_Size) * _I + (2 + PML_Size) * _J + 2 + PML_Size;

                _vp = v_p[index_2];
                _vs = v_s[index_2];
                rho = Rho[index_2];

                for (int k = 0; k < 2 + PML_Size; ++k)
                {
                    v_p[index_1] = _vp;
                    v_s[index_1] = _vs;
                    Rho[index_1] = rho;
                    ++index_1;
                }

                for (int k = 2 + PML_Size; k < I + 2 + PML_Size; ++k)
                {
                    v_p[index_1] = v_p[index_2];
                    v_s[index_1] = v_s[index_2];
                    Rho[index_1] = Rho[index_2];
                    ++index_1;
                    ++index_2;
                }

                --index_2;

                _vp = v_p[index_2];
                _vs = v_s[index_2];
                rho = Rho[index_2];

                for (int k = I + 2 + PML_Size; k < 4 + I + 2 * PML_Size; ++k)
                {
                    v_p[index_1] = _vp;
                    v_s[index_1] = _vs;
                    Rho[index_1] = rho;
                    ++index_1;
                }
            }

            for (int j = 2 + PML_Size; j < _size_j - 2; ++j)
            {
                index_1 = i * _I + j * _J;
                index_2 = (_size_i - 3 - PML_Size) * _I + j * _J + 2 + PML_Size;

                _vp = v_p[index_2];
                _vs = v_s[index_2];
                rho = Rho[index_2];

                for (int k = 0; k < 2 + PML_Size; ++k)
                {
                    v_p[index_1] = _vp;
                    v_s[index_1] = _vs;
                    Rho[index_1] = rho;
                    ++index_1;
                }

                for (int k = 2 + PML_Size; k < I + 2 + PML_Size; ++k)
                {
                    v_p[index_1] = v_p[index_2];
                    v_s[index_1] = v_s[index_2];
                    Rho[index_1] = Rho[index_2];
                    ++index_1;
                    ++index_2;
                }

                --index_2;
                _vp = v_p[index_2];
                _vs = v_s[index_2];
                rho = Rho[index_2];

                for (int k = I + 2 + PML_Size; k < 4 + I + 2 * PML_Size; ++k)
                {
                    v_p[index_1] = _vp;
                    v_s[index_1] = _vs;
                    Rho[index_1] = rho;
                    ++index_1;
                }
            }
        }
    }

    /////////////////////////////////////////////////////////////
    // 
    // +++++++++
    // ****** ++
    // ****** ++
    // ****** ++
    //
    /////////////////////////////////////////////////////////////
    if ((rank_coords[0] == 0) && (rank_coords[1] == dims[1] - 1))
    {
        for (int i = 0; i < 2 + PML_Size; ++i)
        {
            for (int j = 2; j < _size_j - PML_Size - 2; ++j)
            {
                index_1 = i * _I + j * _J;
                index_2 = (2 + PML_Size) * _I + j * _J + 2 + PML_Size;

                _vp = v_p[index_2];
                _vs = v_s[index_2];
                rho = Rho[index_2];

                for (int k = 0; k < 2 + PML_Size; ++k)
                {
                    v_p[index_1] = _vp;
                    v_s[index_1] = _vs;
                    Rho[index_1] = rho;
                    ++index_1;
                }

                for (int k = 2 + PML_Size; k < I + 2 + PML_Size; ++k)
                {
                    v_p[index_1] = v_p[index_2];
                    v_s[index_1] = v_s[index_2];
                    Rho[index_1] = Rho[index_2];
                    ++index_1;
                    ++index_2;
                }

                --index_2;
                _vp = v_p[index_2];
                _vs = v_s[index_2];
                rho = Rho[index_2];

                for (int k = I + 2 + PML_Size; k < 4 + I + 2 * PML_Size; ++k)
                {
                    v_p[index_1] = _vp;
                    v_s[index_1] = _vs;
                    Rho[index_1] = rho;
                    ++index_1;
                }
            }

            for (int j = _size_j - PML_Size - 2; j < _size_j; ++j)
            {
                index_1 = i * _I + j * _J;
                index_2 = (2 + PML_Size) * _I + (_size_j - 3 - PML_Size) * _J + 2 + PML_Size;

                _vp = v_p[index_2];
                _vs = v_s[index_2];
                rho = Rho[index_2];

                for (int k = 0; k < 2 + PML_Size; ++k)
                {
                    v_p[index_1] = _vp;
                    v_s[index_1] = _vs;
                    Rho[index_1] = rho;
                    ++index_1;
                }

                for (int k = 2 + PML_Size; k < I + 2 + PML_Size; ++k)
                {
                    v_p[index_1] = v_p[index_2];
                    v_s[index_1] = v_s[index_2];
                    Rho[index_1] = Rho[index_2];
                    ++index_1;
                    ++index_2;
                }

                --index_2;
                _vp = v_p[index_2];
                _vs = v_s[index_2];
                rho = Rho[index_2];

                for (int k = I + 2 + PML_Size; k < 4 + I + 2 * PML_Size; ++k)
                {
                    v_p[index_1] = _vp;
                    v_s[index_1] = _vs;
                    Rho[index_1] = rho;
                    ++index_1;
                }
            }
        }

        for (int i = 2 + PML_Size; i < _size_i - 2; ++i)
        {
            for (int j = 2; j < _size_j - PML_Size - 2; ++j)
            {
                index_1 = i * _I + j * _J;
                index_2 = i * _I + j * _J + 2 + PML_Size;

                _vp = v_p[index_2];
                _vs = v_s[index_2];
                rho = Rho[index_2];

                for (int k = 0; k < 2 + PML_Size; ++k)
                {
                    v_p[index_1] = _vp;
                    v_s[index_1] = _vs;
                    Rho[index_1] = rho;
                    ++index_1;
                }

                index_1 += I;
                index_2 += I - 1;
                _vp = v_p[index_2];
                _vs = v_s[index_2];
                rho = Rho[index_2];

                for (int k = I + 2 + PML_Size; k < 4 + I + 2 * PML_Size; ++k)
                {
                    v_p[index_1] = _vp;
                    v_s[index_1] = _vs;
                    Rho[index_1] = rho;
                    ++index_1;
                }
            }

            for (int j = _size_j - PML_Size - 2; j < _size_j; ++j)
            {
                index_1 = i * _I + j * _J;
                index_2 = i * _I + (_size_j - 3 - PML_Size) * _J + 2 + PML_Size;

                _vp = v_p[index_2];
                _vs = v_s[index_2];
                rho = Rho[index_2];

                for (int k = 0; k < 2 + PML_Size; ++k)
                {
                    v_p[index_1] = _vp;
                    v_s[index_1] = _vs;
                    Rho[index_1] = rho;
                    ++index_1;
                }

                for (int k = 2 + PML_Size; k < I + 2 + PML_Size; ++k)
                {
                    v_p[index_1] = v_p[index_2];
                    v_s[index_1] = v_s[index_2];
                    Rho[index_1] = Rho[index_2];
                    ++index_1;
                    ++index_2;
                }

                --index_2;
                _vp = v_p[index_2];
                _vs = v_s[index_2];
                rho = Rho[index_2];

                for (int k = I + 2 + PML_Size; k < 4 + I + 2 * PML_Size; ++k)
                {
                    v_p[index_1] = _vp;
                    v_s[index_1] = _vs;
                    Rho[index_1] = rho;
                    ++index_1;
                }
            }
        }
    }

    /////////////////////////////////////////////////////////////
    // 
    // ****** ++
    // ****** ++
    // ****** ++
    // +++++++++
    //
    /////////////////////////////////////////////////////////////
    if ((rank_coords[0] == dims[0] - 1) && (rank_coords[1] == dims[1] - 1))
    {
        for (int i = 2; i < _size_i - PML_Size - 2; ++i)
        {
            for (int j = 2; j < _size_j - PML_Size - 2; ++j)
            {
                index_1 = i * _I + j * _J;
                index_2 = i * _I + j * _J + 2 + PML_Size;

                _vp = v_p[index_2];
                _vs = v_s[index_2];
                rho = Rho[index_2];

                for (int k = 0; k < 2 + PML_Size; ++k)
                {
                    v_p[index_1] = _vp;
                    v_s[index_1] = _vs;
                    Rho[index_1] = rho;
                    ++index_1;
                }

                index_1 += I;
                index_2 += I - 1;
                _vp = v_p[index_2];
                _vs = v_s[index_2];
                rho = Rho[index_2];

                for (int k = I + 2 + PML_Size; k < 4 + I + 2 * PML_Size; ++k)
                {
                    v_p[index_1] = _vp;
                    v_s[index_1] = _vs;
                    Rho[index_1] = rho;
                    ++index_1;
                }
            }

            for (int j = _size_j - PML_Size - 2; j < _size_j; ++j)
            {
                index_1 = i * _I + j * _J;
                index_2 = i * _I + (_size_j - 3 - PML_Size) * _J + 2 + PML_Size;

                _vp = v_p[index_2];
                _vs = v_s[index_2];
                rho = Rho[index_2];

                for (int k = 0; k < 2 + PML_Size; ++k)
                {
                    v_p[index_1] = _vp;
                    v_s[index_1] = _vs;
                    Rho[index_1] = rho;
                    ++index_1;
                }

                for (int k = 2 + PML_Size; k < I + 2 + PML_Size; ++k)
                {
                    v_p[index_1] = v_p[index_2];
                    v_s[index_1] = v_s[index_2];
                    Rho[index_1] = Rho[index_2];
                    ++index_1;
                    ++index_2;
                }

                --index_2;
                _vp = v_p[index_2];
                _vs = v_s[index_2];
                rho = Rho[index_2];

                for (int k = I + 2 + PML_Size; k < 4 + I + 2 * PML_Size; ++k)
                {
                    v_p[index_1] = _vp;
                    v_s[index_1] = _vs;
                    Rho[index_1] = rho;
                    ++index_1;
                }
            }
        }

        for (int i = _size_i - PML_Size - 2; i < _size_i; ++i)
        {
            for (int j = 2; j < _size_j - PML_Size - 2; ++j)
            {
                index_1 = i * _I + j * _J;
                index_2 = (_size_i - 3 - PML_Size) * _I + j * _J + 2 + PML_Size;

                _vp = v_p[index_2];
                _vs = v_s[index_2];
                rho = Rho[index_2];

                for (int k = 0; k < 2 + PML_Size; ++k)
                {
                    v_p[index_1] = _vp;
                    v_s[index_1] = _vs;
                    Rho[index_1] = rho;
                    ++index_1;
                }

                for (int k = 2 + PML_Size; k < I + 2 + PML_Size; ++k)
                {
                    v_p[index_1] = v_p[index_2];
                    v_s[index_1] = v_s[index_2];
                    Rho[index_1] = Rho[index_2];
                    ++index_1;
                    ++index_2;
                }

                --index_2;
                _vp = v_p[index_2];
                _vs = v_s[index_2];
                rho = Rho[index_2];

                for (int k = I + 2 + PML_Size; k < 4 + I + 2 * PML_Size; ++k)
                {
                    v_p[index_1] = _vp;
                    v_s[index_1] = _vs;
                    Rho[index_1] = rho;
                    ++index_1;
                }
            }

            for (int j = _size_j - PML_Size - 2; j < _size_j; ++j)
            {
                index_1 = i * _I + j * _J;
                index_2 = (_size_i - 3 - PML_Size) * _I + (_size_j - 3 - PML_Size) * _J + 2 + PML_Size;

                _vp = v_p[index_2];
                _vs = v_s[index_2];
                rho = Rho[index_2];

                for (int k = 0; k < 2 + PML_Size; ++k)
                {
                    v_p[index_1] = _vp;
                    v_s[index_1] = _vs;
                    Rho[index_1] = rho;
                    ++index_1;
                }

                for (int k = 2 + PML_Size; k < I + 2 + PML_Size; ++k)
                {
                    v_p[index_1] = v_p[index_2];
                    v_s[index_1] = v_s[index_2];
                    Rho[index_1] = Rho[index_2];
                    ++index_1;
                    ++index_2;
                }

                --index_2;
                _vp = v_p[index_2];
                _vs = v_s[index_2];
                rho = Rho[index_2];

                for (int k = I + 2 + PML_Size; k < 4 + I + 2 * PML_Size; ++k)
                {
                    v_p[index_1] = _vp;
                    v_s[index_1] = _vs;
                    Rho[index_1] = rho;
                    ++index_1;
                }
            }
        }
    }

    /////////////////////////////////////////////////////////////
    // 
    // ++ ******
    // ++ ******
    // ++ ******
    // ++ ******
    //
    /////////////////////////////////////////////////////////////
    if ((rank_coords[0] != 0) && (rank_coords[0] != dims[0] - 1) && (rank_coords[1] == 0))
    {
        for (int i = 2; i < _size_i - 2; ++i)
        {
            for (int j = 0; j < 2 + PML_Size; ++j)
            {
                index_1 = i * _I + j * _J;
                index_2 = i * _I + (2 + PML_Size) * _J + 2 + PML_Size;

                _vp = v_p[index_2];
                _vs = v_s[index_2];
                rho = Rho[index_2];

                for (int k = 0; k < 2 + PML_Size; ++k)
                {
                    v_p[index_1] = _vp;
                    v_s[index_1] = _vs;
                    Rho[index_1] = rho;
                    ++index_1;
                }

                for (int k = 2 + PML_Size; k < I + 2 + PML_Size; ++k)
                {
                    v_p[index_1] = v_p[index_2];
                    v_s[index_1] = v_s[index_2];
                    Rho[index_1] = Rho[index_2];
                    ++index_1;
                    ++index_2;
                }

                --index_2;
                _vp = v_p[index_2];
                _vs = v_s[index_2];
                rho = Rho[index_2];

                for (int k = I + 2 + PML_Size; k < 4 + I + 2 * PML_Size; ++k)
                {
                    v_p[index_1] = _vp;
                    v_s[index_1] = _vs;
                    Rho[index_1] = rho;
                    ++index_1;
                }
            }

            for (int j = 2 + PML_Size; j < _size_j - 2; ++j)
            {
                index_1 = i * _I + j * _J;
                index_2 = i * _I + j * _J + 2 + PML_Size;

                _vp = v_p[index_2];
                _vs = v_s[index_2];
                rho = Rho[index_2];

                for (int k = 0; k < 2 + PML_Size; ++k)
                {
                    v_p[index_1] = _vp;
                    v_s[index_1] = _vs;
                    Rho[index_1] = rho;
                    ++index_1;
                }

                index_1 += I;
                index_2 += I - 1;
                _vp = v_p[index_2];
                _vs = v_s[index_2];
                rho = Rho[index_2];

                for (int k = I + 2 + PML_Size; k < 4 + I + 2 * PML_Size; ++k)
                {
                    v_p[index_1] = _vp;
                    v_s[index_1] = _vs;
                    Rho[index_1] = rho;
                    ++index_1;
                }
            }
        }
    }

    /////////////////////////////////////////////////////////////
    // 
    // ****** ++
    // ****** ++ 
    // ****** ++
    // ****** ++
    //
    /////////////////////////////////////////////////////////////
    if ((rank_coords[0] != 0) && (rank_coords[0] != dims[0] - 1) && (rank_coords[1] == dims[1] - 1))
    {
        for (int i = 2; i < _size_i - 2; ++i)
        {
            for (int j = 2; j < _size_j - PML_Size - 2; ++j)
            {
                index_1 = i * _I + j * _J;
                index_2 = i * _I + j * _J + 2 + PML_Size;

                _vp = v_p[index_2];
                _vs = v_s[index_2];
                rho = Rho[index_2];

                for (int k = 0; k < 2 + PML_Size; ++k)
                {
                    v_p[index_1] = _vp;
                    v_s[index_1] = _vs;
                    Rho[index_1] = rho;
                    ++index_1;
                }

                index_1 += I;
                index_2 += I - 1;
                _vp = v_p[index_2];
                _vs = v_s[index_2];
                rho = Rho[index_2];

                for (int k = I + 2 + PML_Size; k < 4 + I + 2 * PML_Size; ++k)
                {
                    v_p[index_1] = _vp;
                    v_s[index_1] = _vs;
                    Rho[index_1] = rho;
                    ++index_1;
                }
            }

            for (int j = _size_j - PML_Size - 2; j < _size_j; ++j)
            {
                index_1 = i * _I + j * _J;
                index_2 = i * _I + (_size_j - 3 - PML_Size) * _J + 2 + PML_Size;

                _vp = v_p[index_2];
                _vs = v_s[index_2];
                rho = Rho[index_2];

                for (int k = 0; k < 2 + PML_Size; ++k)
                {
                    v_p[index_1] = _vp;
                    v_s[index_1] = _vs;
                    Rho[index_1] = rho;
                    ++index_1;
                }

                for (int k = 2 + PML_Size; k < I + 2 + PML_Size; ++k)
                {
                    v_p[index_1] = v_p[index_2];
                    v_s[index_1] = v_s[index_2];
                    Rho[index_1] = Rho[index_2];
                    ++index_1;
                    ++index_2;
                }

                --index_2;
                _vp = v_p[index_2];
                _vs = v_s[index_2];
                rho = Rho[index_2];

                for (int k = I + 2 + PML_Size; k < 4 + I + 2 * PML_Size; ++k)
                {
                    v_p[index_1] = _vp;
                    v_s[index_1] = _vs;
                    Rho[index_1] = rho;
                    ++index_1;
                }
            }
        }
    }

    /////////////////////////////////////////////////////////////
    // 
    // ++++++++
    // ******** 
    // ********
    // ********
    //
    /////////////////////////////////////////////////////////////
    if ((rank_coords[1] != 0) && (rank_coords[1] != dims[1] - 1) && (rank_coords[0] == 0))
    {
        for (int j = 2; j < _size_j - 2; ++j)
        {
            for (int i = 0; i < 2 + PML_Size; ++i)
            {
                index_1 = i * _I + j * _J;
                index_2 = (2 + PML_Size) * _I + j * _J + 2 + PML_Size;

                _vp = v_p[index_2];
                _vs = v_s[index_2];
                rho = Rho[index_2];

                for (int k = 0; k < 2 + PML_Size; ++k)
                {
                    v_p[index_1] = _vp;
                    v_s[index_1] = _vs;
                    Rho[index_1] = rho;
                    ++index_1;
                }

                for (int k = 2 + PML_Size; k < I + 2 + PML_Size; ++k)
                {
                    v_p[index_1] = v_p[index_2];
                    v_s[index_1] = v_s[index_2];
                    Rho[index_1] = Rho[index_2];
                    ++index_1;
                    ++index_2;
                }

                --index_2;
                _vp = v_p[index_2];
                _vs = v_s[index_2];
                rho = Rho[index_2];

                for (int k = I + 2 + PML_Size; k < 4 + I + 2 * PML_Size; ++k)
                {
                    v_p[index_1] = _vp;
                    v_s[index_1] = _vs;
                    Rho[index_1] = rho;
                    ++index_1;
                }
            }

            for (int i = 2 + PML_Size; i < _size_i - 2; ++i)
            {
                index_1 = i * _I + j * _J;
                index_2 = i * _I + j * _J + 2 + PML_Size;

                _vp = v_p[index_2];
                _vs = v_s[index_2];
                rho = Rho[index_2];

                for (int k = 0; k < 2 + PML_Size; ++k)
                {
                    v_p[index_1] = _vp;
                    v_s[index_1] = _vs;
                    Rho[index_1] = rho;
                    ++index_1;
                }

                index_1 += I;
                index_2 += I - 1;
                _vp = v_p[index_2];
                _vs = v_s[index_2];
                rho = Rho[index_2];

                for (int k = I + 2 + PML_Size; k < 4 + I + 2 * PML_Size; ++k)
                {
                    v_p[index_1] = _vp;
                    v_s[index_1] = _vs;
                    Rho[index_1] = rho;
                    ++index_1;
                }
            }
        }
    }

    /////////////////////////////////////////////////////////////
    // 
    // ******** 
    // ********
    // ********
    // ++++++++
    //
    /////////////////////////////////////////////////////////////
    if ((rank_coords[1] != 0) && (rank_coords[1] != dims[1] - 1) && (rank_coords[0] == dims[0] - 1))
    {
        for (int j = 2; j < _size_j - 2; ++j)
        {
            for (int i = 2; i < _size_i - PML_Size - 2; ++i)
            {
                index_1 = i * _I + j * _J;
                index_2 = i * _I + j * _J + 2 + PML_Size;

                _vp = v_p[index_2];
                _vs = v_s[index_2];
                rho = Rho[index_2];

                for (int k = 0; k < 2 + PML_Size; ++k)
                {
                    v_p[index_1] = _vp;
                    v_s[index_1] = _vs;
                    Rho[index_1] = rho;
                    ++index_1;
                }

                index_1 += I;
                index_2 += I - 1;
                _vp = v_p[index_2];
                _vs = v_s[index_2];
                rho = Rho[index_2];

                for (int k = I + 2 + PML_Size; k < 4 + I + 2 * PML_Size; ++k)
                {
                    v_p[index_1] = _vp;
                    v_s[index_1] = _vs;
                    Rho[index_1] = rho;
                    ++index_1;
                }
            }

            for (int i = _size_i - PML_Size - 2; j < _size_i; ++i)
            {
                index_1 = i * _I + j * _J;
                index_2 = (_size_i - 3 - PML_Size) * _I + j * _J + 2 + PML_Size;

                _vp = v_p[index_2];
                _vs = v_s[index_2];
                rho = Rho[index_2];

                for (int k = 0; k < 2 + PML_Size; ++k)
                {
                    v_p[index_1] = _vp;
                    v_s[index_1] = _vs;
                    Rho[index_1] = rho;
                    ++index_1;
                }

                for (int k = 2 + PML_Size; k < I + 2 + PML_Size; ++k)
                {
                    v_p[index_1] = v_p[index_2];
                    v_s[index_1] = v_s[index_2];
                    Rho[index_1] = Rho[index_2];
                    ++index_1;
                    ++index_2;
                }

                --index_2;
                _vp = v_p[index_2];
                _vs = v_s[index_2];
                rho = Rho[index_2];

                for (int k = I + 2 + PML_Size; k < 4 + I + 2 * PML_Size; ++k)
                {
                    v_p[index_1] = _vp;
                    v_s[index_1] = _vs;
                    Rho[index_1] = rho;
                    ++index_1;
                }
            }
        }
    }


    MPI_Datatype left_subaray_send_param;
    MPI_Datatype left_subaray_recv_param;
    MPI_Datatype right_subaray_send_param;
    MPI_Datatype right_subaray_recv_param;
    MPI_Datatype upper_subaray_send_param;
    MPI_Datatype upper_subaray_recv_param;
    MPI_Datatype lower_subaray_send_param;
    MPI_Datatype lower_subaray_recv_param;

    int gsizes[3];
    int lsizes[3];
    int starts[3];

    gsizes[0] = _size_i;
    gsizes[1] = _size_j;
    gsizes[2] = _size_k;

    // ����� ������� ��� ��������
    lsizes[0] = _size_i - 4;
    lsizes[1] = 2;
    lsizes[2] = _size_k - 4;

    starts[0] = 2;
    starts[1] = 2;
    starts[2] = 2;
    MPI_Type_create_subarray(3, gsizes, lsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &left_subaray_send_param);
    MPI_Type_commit(&left_subaray_send_param);

    //printf("left_subaray_send_param\n    gsizes:\n        %d\n        %d\n        %d\n", gsizes[0], gsizes[1], gsizes[2]);
    //printf("    lsizes:\n        %d\n        %d\n        %d\n", lsizes[0], lsizes[1], lsizes[2]);
    //printf("    starts:\n        %d\n        %d\n        %d\n", starts[0], starts[1], starts[2]);

    // ����� ������� ��� ������
    starts[1] = 0;
    MPI_Type_create_subarray(3, gsizes, lsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &left_subaray_recv_param);
    MPI_Type_commit(&left_subaray_recv_param);

    //printf("left_subaray_recv_param\n    gsizes:\n        %d\n        %d\n        %d\n", gsizes[0], gsizes[1], gsizes[2]);
    //printf("    lsizes:\n        %d\n        %d\n        %d\n", lsizes[0], lsizes[1], lsizes[2]);
    //printf("    starts:\n        %d\n        %d\n        %d\n", starts[0], starts[1], starts[2]);

    // ������ ������� ��� ��������
    starts[1] = _size_j - 4;
    MPI_Type_create_subarray(3, gsizes, lsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &right_subaray_send_param);
    MPI_Type_commit(&right_subaray_send_param);

    //printf("right_subaray_send_param\n    gsizes:\n        %d\n        %d\n        %d\n", gsizes[0], gsizes[1], gsizes[2]);
    //printf("    lsizes:\n        %d\n        %d\n        %d\n", lsizes[0], lsizes[1], lsizes[2]);
    //printf("    starts:\n        %d\n        %d\n        %d\n", starts[0], starts[1], starts[2]);

    // ������ ������� ��� ������
    starts[1] = _size_j - 2;
    MPI_Type_create_subarray(3, gsizes, lsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &right_subaray_recv_param);
    MPI_Type_commit(&right_subaray_recv_param);

    //printf("right_subaray_recv_param\n    gsizes:\n        %d\n        %d\n        %d\n", gsizes[0], gsizes[1], gsizes[2]);
    //printf("    lsizes:\n        %d\n        %d\n        %d\n", lsizes[0], lsizes[1], lsizes[2]);
    //printf("    starts:\n        %d\n        %d\n        %d\n", starts[0], starts[1], starts[2]);


    // ������� ������� ��� ��������
    lsizes[0] = 2;
    lsizes[1] = _size_j - 4;
    lsizes[2] = _size_k - 4;

    starts[0] = 2;
    starts[1] = 2;
    starts[2] = 2;
    MPI_Type_create_subarray(3, gsizes, lsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &upper_subaray_send_param);
    MPI_Type_commit(&upper_subaray_send_param);

    //printf("upper_subaray_send_param\n    gsizes:\n        %d\n        %d\n        %d\n", gsizes[0], gsizes[1], gsizes[2]);
    //printf("    lsizes:\n        %d\n        %d\n        %d\n", lsizes[0], lsizes[1], lsizes[2]);
    //printf("    starts:\n        %d\n        %d\n        %d\n", starts[0], starts[1], starts[2]);

    // ������� ������� ��� ������
    starts[0] = 0;
    MPI_Type_create_subarray(3, gsizes, lsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &upper_subaray_recv_param);
    MPI_Type_commit(&upper_subaray_recv_param);

    //printf("upper_subaray_recv_param\n    gsizes:\n        %d\n        %d\n        %d\n", gsizes[0], gsizes[1], gsizes[2]);
    //printf("    lsizes:\n        %d\n        %d\n        %d\n", lsizes[0], lsizes[1], lsizes[2]);
    //printf("    starts:\n        %d\n        %d\n        %d\n", starts[0], starts[1], starts[2]);

    // ������ ������� ��� ��������
    starts[0] = _size_i - 4;
    MPI_Type_create_subarray(3, gsizes, lsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &lower_subaray_send_param);
    MPI_Type_commit(&lower_subaray_send_param);

    //printf("lower_subaray_send_param\n    gsizes:\n        %d\n        %d\n        %d\n", gsizes[0], gsizes[1], gsizes[2]);
    //printf("    lsizes:\n        %d\n        %d\n        %d\n", lsizes[0], lsizes[1], lsizes[2]);
    //printf("    starts:\n        %d\n        %d\n        %d\n", starts[0], starts[1], starts[2]);

    // ������ ������� ��� ������
    starts[0] = _size_i - 2;
    MPI_Type_create_subarray(3, gsizes, lsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &lower_subaray_recv_param);
    MPI_Type_commit(&lower_subaray_recv_param);

    //printf("lower_subaray_recv_param\n    gsizes:\n        %d\n        %d\n        %d\n", gsizes[0], gsizes[1], gsizes[2]);
    //printf("    lsizes:\n        %d\n        %d\n        %d\n", lsizes[0], lsizes[1], lsizes[2]);
    //printf("    starts:\n        %d\n        %d\n        %d\n", starts[0], starts[1], starts[2]);

    // ������ ������ ����� ��������: ��������� ���������� ����� �� ��� x.
        MPI_Sendrecv(
            v_p, 1, lower_subaray_send_param, lower_rank, 100,
            v_p, 1, upper_subaray_recv_param, upper_rank, 100,
            communicator, MPI_STATUS_IGNORE);
        MPI_Sendrecv(
            v_s, 1, lower_subaray_send_param, lower_rank, 101,
            v_s, 1, upper_subaray_recv_param, upper_rank, 101,
            communicator, MPI_STATUS_IGNORE);
        MPI_Sendrecv(
            Rho, 1, lower_subaray_send_param, lower_rank, 102,
            Rho, 1, upper_subaray_recv_param, upper_rank, 102,
            communicator, MPI_STATUS_IGNORE);

        MPI_Sendrecv(
            v_p, 1, upper_subaray_send_param, upper_rank, 103,
            v_p, 1, lower_subaray_recv_param, lower_rank, 103,
            communicator, MPI_STATUS_IGNORE);
        MPI_Sendrecv(
            v_s, 1, upper_subaray_send_param, upper_rank, 104,
            v_s, 1, lower_subaray_recv_param, lower_rank, 104,
            communicator, MPI_STATUS_IGNORE);
        MPI_Sendrecv(
            Rho, 1, upper_subaray_send_param, upper_rank, 105,
            Rho, 1, lower_subaray_recv_param, lower_rank, 105,
            communicator, MPI_STATUS_IGNORE);

    // ������ ������ ����� ��������: ��������� ���������� ����� �� ��� y.
        MPI_Sendrecv(
            v_p, 1, right_subaray_send_param, right_rank, 106,
            v_p, 1, left_subaray_recv_param, left_rank, 106,
            communicator, MPI_STATUS_IGNORE);
        MPI_Sendrecv(
            v_s, 1, right_subaray_send_param, right_rank, 107,
            v_s, 1, left_subaray_recv_param, left_rank, 107,
            communicator, MPI_STATUS_IGNORE);
        MPI_Sendrecv(
            Rho, 1, right_subaray_send_param, right_rank, 108,
            Rho, 1, left_subaray_recv_param, left_rank, 108,
            communicator, MPI_STATUS_IGNORE);

        MPI_Sendrecv(
            v_p, 1, left_subaray_send_param, left_rank, 109,
            v_p, 1, right_subaray_recv_param, right_rank, 109,
            communicator, MPI_STATUS_IGNORE);
        MPI_Sendrecv(
            v_s, 1, left_subaray_send_param, left_rank, 110,
            v_s, 1, right_subaray_recv_param, right_rank, 110,
            communicator, MPI_STATUS_IGNORE);
        MPI_Sendrecv(
            Rho, 1, left_subaray_send_param, left_rank, 111,
            Rho, 1, right_subaray_recv_param, right_rank, 111,
            communicator, MPI_STATUS_IGNORE);

    MPI_Type_free(&left_subaray_send_param);
    MPI_Type_free(&left_subaray_recv_param);

    MPI_Type_free(&right_subaray_send_param);
    MPI_Type_free(&right_subaray_recv_param);

    MPI_Type_free(&upper_subaray_send_param);
    MPI_Type_free(&upper_subaray_recv_param);

    MPI_Type_free(&lower_subaray_send_param);
    MPI_Type_free(&lower_subaray_recv_param);

    /*for (int i = starti_rw; i < endi_rw; i++)
    {
        for (int j = startj_rw; j < endj_rw; j++)
        {
            for (int k = 2 + PML_Size; k < 2 + I + PML_Size; k++)
            {
                if (v_p[i * _I + j * _J + k] != 3.2)
                    std::cout << v_p[i * _I + j * _J + k] << "\n";
            }
        }
    }*/

    /*for (int i = 0; i < _size_i; i++)
    {
        for (int j = 0; j < _size_j; j++)
        {
            for (int k = 0; k < _size_k; k++)
            {
                if (v_p[i * _I + j * _J + k] != 3.8)
                {
                    std::cout << "vp = " << v_p[i * _I + j * _J + k] << "\n";
                    printf("index = ( %d, %d, %d )\n", i, j, k);
                }
            }
        }
    }*/
}

Wave3d::Wave3d(int I, double T, std::function<double(double, double, double, double)> f)
{
    // ������������� ���������� ��� ������������ ������ ���������
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // ������������ �������. ������� � ���������� ���������
    ierror = MPI_Dims_create(size, 2, dims);
    if (ierror != MPI_SUCCESS)
        printf("MPI_Dims_create error!");

    ierror = MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &communicator);
    if (ierror != MPI_SUCCESS)
        printf("MPI_Cart_create error!");

    ierror = MPI_Comm_rank(communicator, &rank);
    ierror = MPI_Cart_coords(communicator, rank, 2, rank_coords);

    // ����������� ��������� "�����" �  "������"
    ierror = MPI_Cart_shift(communicator, 1, 1, &left_rank, &right_rank);
    if (ierror != MPI_SUCCESS)
        printf("MPI_Cart_shift error!");

    // ����������� ��������� "�����" �  "������"
    ierror = MPI_Cart_shift(communicator, 0, 1, &upper_rank, &lower_rank);
    if (ierror != MPI_SUCCESS)
        printf("MPI_Cart_shift error!");

    ///////////////////////////////////////////////////////
    // ������������ ��������� � ��������� XY
    //    +---+---+---+
    //    | 0 | 1 | 2 |                   upper
    //    +---+---+---+                   +---+
    //    | 3 | 4 | 5 |              left | i | right
    //    +---+---+---+                   +---+
    //    | 6 | 7 | 8 |                   lower
    //    +---+---+---+
    // 
    ///////////////////////////////////////////////////////

    //printf("rank: (%d, %d) = %d:\nleft_rank = %d\nright_rank = %d\nlower_rank = %d\nupper_rank = %d\n", rank_coords[0], rank_coords[1], rank, left_rank, right_rank, lower_rank, upper_rank);

    // �������� ��������� Vars ��� MPI ���������
    const int nitems = 9;
    int          blocklengths[9] = { 1, 1, 1, 1, 1, 1, 1, 1, 1 };
    MPI_Datatype types[9] = { MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE };
    MPI_Aint     offsets[9];

    offsets[0] = offsetof(Vars, u_x);
    offsets[1] = offsetof(Vars, u_y);
    offsets[2] = offsetof(Vars, u_z);
    offsets[3] = offsetof(Vars, sigma_xx);
    offsets[4] = offsetof(Vars, sigma_yy);
    offsets[5] = offsetof(Vars, sigma_zz);
    offsets[6] = offsetof(Vars, sigma_xy);
    offsets[7] = offsetof(Vars, sigma_yz);
    offsets[8] = offsetof(Vars, sigma_zx);

    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpi_vars);
    MPI_Type_commit(&mpi_vars);

    this->f = f;
    this->h = 1. / (I - 1.);
    this->T = T;
    this->I = I;

    // ����������� �������� ����������� ��� ������� ��������
    // �� ����������� x
    int h1 = (I + 2 * PML_Size) / dims[0];
    int m1 = (I + 2 * PML_Size) % dims[0];

    // �� ����������� y
    int h2 = (I + 2 * PML_Size) / dims[1];
    int m2 = (I + 2 * PML_Size) % dims[1];

    if (rank_coords[0] < m1)
        h1++;
    if (rank_coords[1] < m2)
        h2++;

    _size_i = h1 + 4;
    _size_j = h2 + 4;
    _size_k = I + 2 * PML_Size + 4;

    if (rank_coords[0] == 0)
    {
        starti_rw = 2 + PML_Size;
        endi_rw = _size_i - 2;
    }
    else if (rank_coords[0] == dims[0] - 1)
    {
        starti_rw = 2;
        endi_rw = _size_i - PML_Size - 2;
    }
    else
    {
        starti_rw = 2;
        endi_rw = _size_i - 2;
    }

    if (rank_coords[1] == 0)
    {
        startj_rw = 2 + PML_Size;
        endj_rw = _size_j - 2;
    }
    else if (rank_coords[1] == dims[1] - 1)
    {
        startj_rw = 2;
        endj_rw = _size_j - PML_Size - 2;
    }
    else
    {
        startj_rw = 2;
        endj_rw = _size_j - 2;
    }

    // ������� �������� �������������� ������� � �� ������� � ���������� ��������
    int gsizes[3] = { I , I , I};
    int lsizes[3] = { (endi_rw - starti_rw), (endj_rw - startj_rw), I};
    count_rw = lsizes[0] * lsizes[1] * lsizes[2];
    buf_rw = new double[count_rw];

    // �������� �� ���� ������������
    int starts[3];
    // ����������� �������� ��� ������������� ������ / ������
    starts[0] = 0;
    if (rank_coords[0] != 0)
    {
        if (rank_coords[0] < m1)
        {
            starts[0] = ((rank_coords[0] - 1) * h1 + (h1 - PML_Size));
        }
        else
        {
            starts[0] = ((m1 - 1) * (h1 + 1) + (h1 + 1 - PML_Size) + (rank_coords[0] - m1) * h1);
        }
    }
    starts[1] = 0;
    if (rank_coords[1] != 0)
    {
        if (rank_coords[1] < m2)
        {
            starts[1] = ((rank_coords[1] - 1) * h2 + (h2 - PML_Size));
        }
        else
        {
            starts[1] = ((m2 - 1) * (h2 + 1) + (h2 + 1 - PML_Size) + (rank_coords[1] - m2) * h2);
        }
    }
    starts[2] = 0;

    starts_f[0] = starts[0];
    starts_f[1] = starts[1];
    starts_f[2] = starts[2];

    // ���������� ��������� - ���������� � ���������� ������������.
    // subarray_rw - ���������� ����� ��������� ������ � ����� ��� ���������� / ������ ������� ��������.
    MPI_Type_create_subarray(3, gsizes, lsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &subarray_rw);
    MPI_Type_commit(&subarray_rw);

    gsizes[0] = _size_i;
    gsizes[1] = _size_j;
    gsizes[2] = _size_k;

    // ����� ������� ��� ��������
    lsizes[0] = _size_i - 4;
    lsizes[1] = 2;
    lsizes[2] = _size_k - 4;

    starts[0] = 2;
    starts[1] = 2;
    starts[2] = 2;
    MPI_Type_create_subarray(3, gsizes, lsizes, starts, MPI_ORDER_C, mpi_vars, &left_subaray_send);
    MPI_Type_commit(&left_subaray_send);

    // ����� ������� ��� ������
    starts[1] = 0;
    MPI_Type_create_subarray(3, gsizes, lsizes, starts, MPI_ORDER_C, mpi_vars, &left_subaray_recv);
    MPI_Type_commit(&left_subaray_recv);

    // ������ ������� ��� ��������
    starts[1] = _size_j - 4;
    MPI_Type_create_subarray(3, gsizes, lsizes, starts, MPI_ORDER_C, mpi_vars, &right_subaray_send);
    MPI_Type_commit(&right_subaray_send);

    // ������ ������� ��� ������
    starts[1] = _size_j - 2;
    MPI_Type_create_subarray(3, gsizes, lsizes, starts, MPI_ORDER_C, mpi_vars, &right_subaray_recv);
    MPI_Type_commit(&right_subaray_recv);


    // ������� ������� ��� ��������
    lsizes[0] = 2;
    lsizes[1] = _size_j - 4;
    lsizes[2] = _size_k - 4;

    starts[0] = 2;
    starts[1] = 2;
    starts[2] = 2;
    MPI_Type_create_subarray(3, gsizes, lsizes, starts, MPI_ORDER_C, mpi_vars, &upper_subaray_send);
    MPI_Type_commit(&upper_subaray_send);

    // ������� ������� ��� ������
    starts[0] = 0;
    MPI_Type_create_subarray(3, gsizes, lsizes, starts, MPI_ORDER_C, mpi_vars, &upper_subaray_recv);
    MPI_Type_commit(&upper_subaray_recv);

    // ������ ������� ��� ��������
    starts[0] = _size_i - 4;
    MPI_Type_create_subarray(3, gsizes, lsizes, starts, MPI_ORDER_C, mpi_vars, &lower_subaray_send);
    MPI_Type_commit(&lower_subaray_send);

    // ������ ������� ��� ������
    starts[0] = _size_i - 2;
    MPI_Type_create_subarray(3, gsizes, lsizes, starts, MPI_ORDER_C, mpi_vars, &lower_subaray_recv);
    MPI_Type_commit(&lower_subaray_recv);

    count_sendrecv_i = 2 * (_size_j - 4) * (_size_k - 4);
    count_sendrecv_j = (_size_i - 4) * 2 * (_size_k - 4);

    v_p = new double[_size_i * _size_j * _size_k];
    v_s = new double[_size_i * _size_j * _size_k];
    Rho = new double[_size_i * _size_j * _size_k];

    _I = _size_j * _size_k;
    _J = _size_k;

    _read_param_from_file();

    // ����������� ������������� �������� �������� ���������� ���� ��� ������� ���� �� �������
    double max = 0.;
    double* _max = new double[dims[0] * dims[1]];

    long int index = 0;
    for (int i = starti_rw; i < endi_rw; i++)
    {
        for (int j = startj_rw; j < endj_rw; j++)
        {
            for (int k = 2 + PML_Size; k < 4 + I + PML_Size; k++)
            {
                index = i * _I + j * _J + k;
                max = max < v_p[index] ? v_p[index] : max;
            }
        }
    }

    //printf("rank: (%d, %d):\n", rank_coords[0], rank_coords[1]);
    //std::cout << " max rank = " << max << std::endl;
    MPI_Gather(&max, 1, MPI_DOUBLE, _max, 1, MPI_DOUBLE, 0, communicator);

    if (rank == 0)
    {
        for (int i = 0; i < dims[0] * dims[1]; i++)
        {
            max = max < _max[i] ? _max[i] : max;
        }
    }

    MPI_Bcast(&max, 1, MPI_DOUBLE, 0, communicator);
    delete[] _max;

    //printf("rank: (%d, %d):\n", rank_coords[0], rank_coords[1]);

    this->tau = (h / max * 0.95);
    this->N = int(T / (tau * 3));

    //std::cout << " max rank = " << max << std::endl;
    //std::cout << " tau rank = " << tau << std::endl;
    //std::cout << " N rank = " << N << std::endl;

    long int size_data_one_iter = I * I * I * 8; // � ������
    long int size_data = N * size_data_one_iter; // � ������
    this->num_iters_in_file = (int)(this->max_size_file / size_data_one_iter);
    this->num_file = (int)(N / this->num_iters_in_file) + 1;

    w_rank_curr = new Vars[_size_i * _size_j * _size_k];
    w_rank_next = new Vars[_size_i * _size_j * _size_k];

    g_rank_next = new Vars[_size_i * _size_j * _size_k];
    g_rank_curr = new Vars[_size_i * _size_j * _size_k];

    Vars w0 = { 0., 0., 0., 0., 0., 0., 0., 0., 0. };
    index = 0;
    for (int i = 0; i < _size_i; i++)
    {
        for (int j = 0; j < _size_j; j++)
        {
            for (int k = 0; k < _size_k; k++)
            {
                w_rank_curr[index] = w0;
                w_rank_next[index] = w0;
                g_rank_curr[index] = w0;
                g_rank_next[index] = w0;

                ++index;
            }
        }
    }
    _write_to_file_Abs_U();
}

Wave3d::~Wave3d()
{
    delete[] w_rank_curr;
    delete[] w_rank_next;
    delete[] g_rank_curr;
    delete[] g_rank_next;
    delete[] buf_rw;
    delete[] v_p;
    delete[] v_s;
    delete[] Rho;

    MPI_Type_free(&mpi_vars);
    MPI_Type_free(&subarray_rw);

    MPI_Type_free(&left_subaray_send);
    MPI_Type_free(&left_subaray_recv);

    MPI_Type_free(&right_subaray_send);
    MPI_Type_free(&right_subaray_recv);

    MPI_Type_free(&upper_subaray_send);
    MPI_Type_free(&upper_subaray_recv);

    MPI_Type_free(&lower_subaray_send);
    MPI_Type_free(&lower_subaray_recv);
    MPI_Finalize();
}

void Wave3d::solve()
{
    if (rank == 0)
    {   
        std::ofstream param("param.txt");
        param << "N = " << N << "\n";
        param << "I = " << I << "\n";
        param << "J = " << I << "\n";
        param << "K = " << I << "\n";
        param << "T = " << T << "\n";
        param << "NumFile = " << this->num_file << "\n";
        param << "NumIter = " << this->num_iters_in_file << "\n";
        param << "NumiterLast = " << N - (num_file - 1) * num_iters_in_file;
        param.close();
    }

    this->n = 0;
    while (n < N)
    {
        _make_step_X();
        _make_step_Y();
        _make_step_Z();
        F();
        _write_to_file_Abs_U();
        //_write_to_file();
        ++n;
        if (n >= N)
            break;

        _make_step_Y();
        _make_step_X();
        _make_step_Z();
        F();
        _write_to_file_Abs_U();
        //_write_to_file();
        ++n;
        if (n >= N)
            break;

        _make_step_Z();
        _make_step_X();
        _make_step_Y();
        F();
        _write_to_file_Abs_U();
        //_write_to_file();
        ++n;
        if (n >= N)
            break;

        _make_step_X();
        _make_step_Z();
        _make_step_Y();
        F();
        _write_to_file_Abs_U();
        //_write_to_file();
        ++n;
        if (n >= N)
            break;

        _make_step_Y();
        _make_step_Z();
        _make_step_X();
        F();
        _write_to_file_Abs_U();
        //_write_to_file();
        ++n;
        if (n >= N)
            break;

        _make_step_Z();
        _make_step_Y();
        _make_step_X();
        F();
        _write_to_file_Abs_U();
        //_write_to_file();
        ++n;
        if (n >= N)
            break;
    }
}