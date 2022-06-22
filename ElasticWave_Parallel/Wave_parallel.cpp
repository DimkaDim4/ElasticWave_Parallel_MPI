#include "Wave_parallel.h"

void Wave3d::F(const int& n)
{
#pragma omp for
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
                    //w_curr[index].u_x += _f;
                    //w_curr[index].u_y += _f;
                    //w_curr[index].u_z += _f;

                    w_curr[index].sigma_xx += _f;
                    w_curr[index].sigma_yy += _f;
                    w_curr[index].sigma_zz += _f;

                    //w_curr[index].sigma_xy += _f;
                    //w_curr[index].sigma_yz += _f;
                    //w_curr[index].sigma_zx += _f;
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
    Vars g_next;
    double vp = v_p[index];
    double vs = v_s[index];

    double deltam; // delta -1
    double delta0; // delta 0
    double deltap; // delta +1
    double alpha;

    // число Куранта 
    double sigma_vs = vs * tau / h;
    double sigma_vp = vp * tau / h;

    // 1-e уравнение переноса
    // lambda = -vp
    deltam = upp.u_x - up.u_x; // delta -1
    delta0 = up.u_x - u.u_x; // delta 0
    deltap = u.u_x - um.u_x; // delta +1

    if (delta0 != 0.)
    {
        alpha = this->_get_alpha(sigma_vp, deltap / delta0, deltam / delta0);
    }
    else
    {
        // Схема Русанова
        alpha = sigma_vp * (sigma_vp * sigma_vp - 1.) / 6.;
    }

    g_next.u_x = u.u_x + 0.5 * sigma_vp * (delta0 + deltap) + 0.5 * sigma_vp * sigma_vp * (delta0 - deltap) + alpha * (deltam - 2. * delta0 + deltap);

    // в случае если схемы второго-третьего порядка не являются монотоннымы - вычисляем по монотонной схеме первого порядка
    if (alpha == -1.)
    {
        g_next.u_x = sigma_vp * up.u_x + (1. - sigma_vp) * u.u_x;
    }

    // 2-e уравнение переноса
    // lambda = -vs
    deltam = upp.u_y - up.u_y; // delta -1
    delta0 = up.u_y - u.u_y; // delta 0
    deltap = u.u_y - um.u_y; // delta +1

    if (delta0 != 0.)
    {
        alpha = this->_get_alpha(sigma_vs, deltap / delta0, deltam / delta0);
    }
    else
    {
        // Схема Русанова
        alpha = sigma_vs * (sigma_vs * sigma_vs - 1.) / 6.;
    }

    g_next.u_y = u.u_y + 0.5 * sigma_vs * (delta0 + deltap) + 0.5 * sigma_vs * sigma_vs * (delta0 - deltap) + alpha * (deltam - 2. * delta0 + deltap);

    // в случае если схемы второго-третьего порядка не являются монотоннымы - вычисляем по монотонной схеме первого порядка
    if (alpha == -1.)
    {
        g_next.u_y = sigma_vs * up.u_y + (1. - sigma_vs) * u.u_y;
    }

    // 3-e уравнение переноса
    // lambda = -vs
    deltam = upp.u_z - up.u_z; // delta -1
    delta0 = up.u_z - u.u_z; // delta 0
    deltap = u.u_z - um.u_z; // delta +1

    if (delta0 != 0)
    {
        alpha = this->_get_alpha(sigma_vs, deltap / delta0, deltam / delta0);
    }
    else
    {
        // Схема Русанова
        alpha = sigma_vs * (sigma_vs * sigma_vs - 1.) / 6.;
    }

    g_next.u_z = u.u_z + 0.5 * sigma_vs * (delta0 + deltap) + 0.5 * sigma_vs * sigma_vs * (delta0 - deltap) + alpha * (deltam - 2. * delta0 + deltap);

    // в случае если схемы второго-третьего порядка не являются монотоннымы - вычисляем по монотонной схеме первого порядка
    if (alpha == -1.)
    {
        g_next.u_z = sigma_vs * up.u_z + (1. - sigma_vs) * u.u_z;
    }

    // 4-e уравнение переноса
    // lambda = 0
    g_next.sigma_xx = u.sigma_xx;

    // 5-e уравнение переноса
    // lambda = 0
    g_next.sigma_yy = u.sigma_yy;

    // 6-e уравнение переноса
    // lambda = 0
    g_next.sigma_zz = u.sigma_zz;

    // 7e уравнение переноса
    // lambda = vs
    deltam = umm.sigma_xy - um.sigma_xy; // delta -1
    delta0 = um.sigma_xy - u.sigma_xy; // delta 0
    deltap = u.sigma_xy - up.sigma_xy; // delta +1

    if (delta0 != 0)
    {
        alpha = this->_get_alpha(sigma_vs, deltap / delta0, deltam / delta0);
    }
    else
    {
        // Схема Русанова
        alpha = sigma_vs * (sigma_vs * sigma_vs - 1.) / 6.;
    }

    g_next.sigma_xy = u.sigma_xy + 0.5 * sigma_vs * (delta0 + deltap) + 0.5 * sigma_vs * sigma_vs * (delta0 - deltap) + alpha * (deltam - 2. * delta0 + deltap);

    // в случае если схемы второго-третьего порядка не являются монотоннымы - вычисляем по монотонной схеме первого порядка
    if (alpha == -1.)
    {
        g_next.sigma_xy = sigma_vs * um.sigma_xy + (1. - sigma_vs) * u.sigma_xy;
    }

    // 8e уравнение переноса
    // lambda = vs
    //sigma = vs * tau / h;
    deltam = umm.sigma_yz - um.sigma_yz; // delta -1
    delta0 = um.sigma_yz - u.sigma_yz; // delta 0
    deltap = u.sigma_yz - up.sigma_yz; // delta +1

    if (delta0 != 0)
    {
        alpha = this->_get_alpha(sigma_vs, deltap / delta0, deltam / delta0);
    }
    else
    {
        // Схема Русанова
        alpha = sigma_vs * (sigma_vs * sigma_vs - 1.) / 6.;
    }

    g_next.sigma_yz = u.sigma_yz + 0.5 * sigma_vs * (delta0 + deltap) + 0.5 * sigma_vs * sigma_vs * (delta0 - deltap) + alpha * (deltam - 2. * delta0 + deltap);

    // в случае если схемы второго-третьего порядка не являются монотоннымы - вычисляем по монотонной схеме первого порядка
    if (alpha == -1.)
    {
        g_next.sigma_yz = sigma_vs * um.sigma_yz + (1. - sigma_vs) * u.sigma_yz;
    }

    // 9e уравнение переноса
    // lambda = vp
    deltam = umm.sigma_zx - um.sigma_zx; // delta -1
    delta0 = um.sigma_zx - u.sigma_zx; // delta 0
    deltap = u.sigma_zx - up.sigma_zx; // delta +1

    if (delta0 != 0)
    {
        alpha = this->_get_alpha(sigma_vp, deltap / delta0, deltam / delta0);
    }
    else
    {
        // Схема Русанова
        alpha = sigma_vp * (sigma_vp * sigma_vp - 1.) / 6.;
    }

    g_next.sigma_zx = u.sigma_zx + 0.5 * sigma_vp * (delta0 + deltap) + 0.5 * sigma_vp * sigma_vp * (delta0 - deltap) + alpha * (deltam - 2. * delta0 + deltap);

    // в случае если схемы второго-третьего порядка не являются монотоннымы - вычисляем по монотонной схеме первого порядка
    if (alpha == -1.)
    {
        g_next.sigma_zx = sigma_vp * um.sigma_zx + (1. - sigma_vp) * u.sigma_zx;
    }

    g_rank_next[index] = g_next;
}

void Wave3d::_PML_solve_system_eq(const long int& index, const Vars& umm, const Vars& um, const Vars& u, const Vars& up, const Vars& upp, const double& demp)
{
    Vars g_next;
    double vp = v_p[index];
    double vs = v_s[index];

    double deltam; // delta -1
    double delta0; // delta 0
    double deltap; // delta +1
    double alpha;

    // число Куранта 
    double sigma_vs = vs * tau / h;
    double sigma_vp = vp * tau / h;

    double coeff = 1. - tau * demp;

    // 1-e уравнение переноса
    // lambda = -vp
    deltam = upp.u_x - up.u_x; // delta -1
    delta0 = up.u_x - u.u_x; // delta 0
    deltap = u.u_x - um.u_x; // delta +1

    if (delta0 != 0)
    {
        alpha = this->_PML_get_alpha(sigma_vp, deltap / delta0, deltam / delta0);
    }
    else
    {
        // Схема Русанова
        alpha = sigma_vp * (sigma_vp * sigma_vp - 1.) / 6.;
    }
    
    switch (this->scheme_type)
    {
    case 0: // Схема Лакса-Вендорффа
        g_next.u_x = u.u_x * coeff
            + 0.5 * sigma_vp * (delta0 * coeff + deltap) + 0.5 * sigma_vp * sigma_vp * (delta0 - deltap) + alpha * (deltam - 2. * delta0 + deltap);
        break;
    case 1: // Схема Бима-Уорминга
        g_next.u_x = u.u_x * (1. - 1.5 * tau * demp)
            + 0.5 * sigma_vp * (delta0 * (1. - 2. * tau * demp) + deltap) + 0.5 * sigma_vp * sigma_vp * (delta0 - deltap) + alpha * (deltam - 2. * delta0 + deltap);
        break;
    case 2: // Схема Руснова
        g_next.u_x = u.u_x * coeff
            + 0.5 * sigma_vp * (delta0 + deltap) + 0.5 * sigma_vp * sigma_vp * (delta0 - deltap) + alpha * (deltam - 2. * delta0 + deltap)
            - 0.25 * sigma_vp * tau * demp * (delta0 + deltap) - tau * sigma_vp * sigma_vp * demp * (deltam - deltap) / 12.;
        break;
    case -1:
        g_next.u_x = sigma_vp * up.u_x + (1. - tau * demp - sigma_vp) * u.u_x;
        break;
    }

    // 2-e уравнение переноса
    // lambda = -vs
    deltam = upp.u_y - up.u_y; // delta -1
    delta0 = up.u_y - u.u_y; // delta 0
    deltap = u.u_y - um.u_y; // delta +1

    if (delta0 != 0)
    {
        alpha = this->_PML_get_alpha(sigma_vs, deltap / delta0, deltam / delta0);
    }
    else
    {
        // Схема Русанова
        this->scheme_type = 2;
        alpha = sigma_vs * (sigma_vs * sigma_vs - 1.) / 6.;
    }

    switch (this->scheme_type)
    {
    case 0: // Схема Лакса-Вендорффа
        g_next.u_y = u.u_y * coeff
            + 0.5 * sigma_vs * (delta0 * coeff + deltap) + 0.5 * sigma_vs * sigma_vs * (delta0 - deltap) + alpha * (deltam - 2. * delta0 + deltap);
        break;
    case 1: // Схема Бима-Уорминга
        g_next.u_y = u.u_y * (1. - 1.5 * tau * demp)
            + 0.5 * sigma_vs * (delta0 * (1. - 2. * tau * demp) + deltap) + 0.5 * sigma_vs * sigma_vs * (delta0 - deltap) + alpha * (deltam - 2. * delta0 + deltap);
        break;
    case 2: // Схема Руснова
        g_next.u_y = u.u_y * coeff
            + 0.5 * sigma_vs * (delta0 + deltap) + 0.5 * sigma_vs * sigma_vs * (delta0 - deltap) + alpha * (deltam - 2. * delta0 + deltap)
            - 0.25 * sigma_vs * tau * demp * (delta0 + deltap) - tau * sigma_vs * sigma_vs * demp * (deltam - deltap) / 12.;
        break;
    case -1:
        g_next.u_y = sigma_vs * up.u_y + (1. - tau * demp - sigma_vs) * u.u_y;
        break;
    }

    // 3-e уравнение переноса
    // lambda = -vs
    //sigma = vs * tau / h;
    deltam = upp.u_z - up.u_z; // delta -1
    delta0 = up.u_z - u.u_z; // delta 0
    deltap = u.u_z - um.u_z; // delta +1

    if (delta0 != 0)
    {
        alpha = this->_PML_get_alpha(sigma_vs, deltap / delta0, deltam / delta0);
    }
    else
    {
        // Схема Русанова
        this->scheme_type = 2;
        alpha = sigma_vs * (sigma_vs * sigma_vs - 1.) / 6.;
    }

    switch (this->scheme_type)
    {
    case 0: // Схема Лакса-Вендорффа
        g_next.u_z = u.u_z * coeff
            + 0.5 * sigma_vs * (delta0 * coeff + deltap) + 0.5 * sigma_vs * sigma_vs * (delta0 - deltap) + alpha * (deltam - 2. * delta0 + deltap);
        break;
    case 1: // Схема Бима-Уорминга
        g_next.u_z = u.u_z * (1. - 1.5 * tau * demp)
            + 0.5 * sigma_vs * (delta0 * (1. - 2. * tau * demp) + deltap) + 0.5 * sigma_vs * sigma_vs * (delta0 - deltap) + alpha * (deltam - 2. * delta0 + deltap);
        break;
    case 2: // Схема Руснова
        g_next.u_z = u.u_z * coeff
            + 0.5 * sigma_vs * (delta0 + deltap) + 0.5 * sigma_vs * sigma_vs * (delta0 - deltap) + alpha * (deltam - 2. * delta0 + deltap)
            - 0.25 * sigma_vs * tau * demp * (delta0 + deltap) - tau * sigma_vs * sigma_vs * demp * (deltam - deltap) / 12.;
        break;
    case -1:
        g_next.u_z = sigma_vp * up.u_z + (1. - tau * demp - sigma_vp) * u.u_z;
        break;
    }

    // 4-e уравнение переноса
    // lambda = 0
    g_next.sigma_xx = u.sigma_xx * coeff;

    // 5-e уравнение переноса
    // lambda = 0
    g_next.sigma_yy = u.sigma_yy * coeff;

    // 6-e уравнение переноса
    // lambda = 0
    g_next.sigma_zz = u.sigma_zz * coeff;

    // 7e уравнение переноса
    // lambda = vs
    deltam = umm.sigma_xy - um.sigma_xy; // delta -1
    delta0 = um.sigma_xy - u.sigma_xy; // delta 0
    deltap = u.sigma_xy - up.sigma_xy; // delta +1

    if (delta0 != 0)
    {
        alpha = this->_PML_get_alpha(sigma_vs, deltap / delta0, deltam / delta0);
    }
    else
    {
        // Схема Русанова
        this->scheme_type = 2;
        alpha = sigma_vs * (sigma_vs * sigma_vs - 1.) / 6.;
    }

    switch (this->scheme_type)
    {
    case 0: // Схема Лакса-Вендорффа
        g_next.sigma_xy = u.sigma_xy * coeff
            + 0.5 * sigma_vs * (delta0 * coeff + deltap) + 0.5 * sigma_vs * sigma_vs * (delta0 - deltap) + alpha * (deltam - 2. * delta0 + deltap);
        break;
    case 1: // Схема Бима-Уорминга
        g_next.sigma_xy = u.sigma_xy * (1. - 1.5 * tau * demp)
            + 0.5 * sigma_vs * (delta0 * (1. - 2. * tau * demp) + deltap) + 0.5 * sigma_vs * sigma_vs * (delta0 - deltap) + alpha * (deltam - 2. * delta0 + deltap);
        break;
    case 2: // Схема Руснова
        g_next.sigma_xy = u.sigma_xy * coeff
            + 0.5 * sigma_vs * (delta0 + deltap) + 0.5 * sigma_vs * sigma_vs * (delta0 - deltap) + alpha * (deltam - 2. * delta0 + deltap)
            - 0.25 * sigma_vs * tau * demp * (delta0 + deltap) - tau * sigma_vs * sigma_vs * demp * (deltam - deltap) / 12.;
        break;
    case -1:
        g_next.sigma_xy = sigma_vs * um.sigma_xy + (1. - tau * demp - sigma_vs) * u.sigma_xy;
        break;
    }

    // 8e уравнение переноса
    // lambda = vs
    //sigma = vs * tau / h;
    deltam = umm.sigma_yz - um.sigma_yz; // delta -1
    delta0 = um.sigma_yz - u.sigma_yz; // delta 0
    deltap = u.sigma_yz - up.sigma_yz; // delta +1

    if (delta0 != 0)
    {
        alpha = this->_PML_get_alpha(sigma_vs, deltap / delta0, deltam / delta0);
    }
    else
    {
        // Схема Русанова
        this->scheme_type = 2;
        alpha = sigma_vs * (sigma_vs * sigma_vs - 1.) / 6.;
    }

    switch (this->scheme_type)
    {
    case 0: // Схема Лакса-Вендорффа
        g_next.sigma_yz = u.sigma_yz * coeff
            + 0.5 * sigma_vs * (delta0 * coeff + deltap) + 0.5 * sigma_vs * sigma_vs * (delta0 - deltap) + alpha * (deltam - 2. * delta0 + deltap);
        break;
    case 1: // Схема Бима-Уорминга
        g_next.sigma_yz = u.sigma_yz * (1. - 1.5 * tau * demp)
            + 0.5 * sigma_vs * (delta0 * (1. - 2. * tau * demp) + deltap) + 0.5 * sigma_vs * sigma_vs * (delta0 - deltap) + alpha * (deltam - 2. * delta0 + deltap);
        break;
    case 2: // Схема Руснова
        g_next.sigma_yz = u.sigma_yz * coeff
            + 0.5 * sigma_vs * (delta0 + deltap) + 0.5 * sigma_vs * sigma_vs * (delta0 - deltap) + alpha * (deltam - 2. * delta0 + deltap)
            - 0.25 * sigma_vs * tau * demp * (delta0 + deltap) - tau * sigma_vs * sigma_vs * demp * (deltam - deltap) / 12.;
        break;
    case -1:
        g_next.sigma_yz = sigma_vs * um.sigma_yz + (1. - tau * demp - sigma_vs) * u.sigma_yz;
        break;
    }

    // 9e уравнение переноса
    // lambda = vp
    deltam = umm.sigma_zx - um.sigma_zx; // delta -1
    delta0 = um.sigma_zx - u.sigma_zx; // delta 0
    deltap = u.sigma_zx - up.sigma_zx; // delta +1

    if (delta0 != 0.)
    {
        alpha = this->_PML_get_alpha(sigma_vp, deltap / delta0, deltam / delta0);
    }
    else
    {
        // Схема Русанова
        this->scheme_type = 2;
        alpha = sigma_vp * (sigma_vp * sigma_vp - 1.) / 6.;
    }

    switch (this->scheme_type)
    {
    case 0: // Схема Лакса-Вендорффа
        g_next.sigma_zx = u.sigma_zx * coeff
            + 0.5 * sigma_vp * (delta0 * coeff + deltap) + 0.5 * sigma_vp * sigma_vp * (delta0 - deltap) + alpha * (deltam - 2. * delta0 + deltap);
        break;
    case 1: // Схема Бима-Уорминга
        g_next.sigma_zx = u.sigma_zx * (1. - 1.5 * tau * demp)
            + 0.5 * sigma_vp * (delta0 * (1. - 2. * tau * demp) + deltap) + 0.5 * sigma_vp * sigma_vp * (delta0 - deltap) + alpha * (deltam - 2. * delta0 + deltap);
        break;
    case 2: // Схема Руснова
        g_next.sigma_zx = u.sigma_zx * coeff
            + 0.5 * sigma_vp * (delta0 + deltap) + 0.5 * sigma_vp * sigma_vp * (delta0 - deltap) + alpha * (deltam - 2. * delta0 + deltap)
            - 0.25 * sigma_vp * tau * demp * (delta0 + deltap) - tau * sigma_vp * sigma_vp * demp * (deltam - deltap) / 12.;
        break;
    case -1:
        g_next.sigma_zx = sigma_vp * um.sigma_zx + (1. - tau * demp - sigma_vp) * u.sigma_zx;
        break;
    }

    g_rank_next[index] = g_next;
}

double Wave3d::_get_alpha(const double& sigma, const double& _deltap, const double& _deltam)
{
    double coeff, w;

    coeff = sigma * (sigma * sigma - 1.) / 6.;
    w = 0.5 * sigma * (1. + _deltap) + 0.5 * sigma * sigma * (1. - _deltap) + coeff * (_deltam + _deltap - 2.);
    if ((w >= 0.) && (w <= 1.))
    {
        //printf("3");
        return coeff; // Схема Русанова
    }

    w = sigma * (1. + sigma) * 0.5 + _deltap * sigma * (1. - sigma) * 0.5;
    if ((w >= 0.) && (w <= 1.))
    {
        //printf("2");
        return 0.; // Схема Лакса-Вендроффа
    }

    w = sigma * (3. - sigma) * 0.5 + _deltam * sigma * (sigma - 1.) * 0.5;
    if ((w >= 0.) && (w <= 1.))
    {
        //printf("2");
        return 0.5 * sigma * (sigma - 1.); // Схема Бима-Уорминга
    }

    //printf("1");
    return -1.;
    //return sigma * (sigma * sigma - 1.) / 6.;
}

double Wave3d::_PML_get_alpha(const double& sigma, const double& _deltap, const double& _deltam)
{
    double coeff, w;
    coeff = sigma * (sigma * sigma - 1.) / 6.;
    w = 0.5 * sigma * (1. + _deltap) + 0.5 * sigma * sigma * (1. - _deltap) + coeff * (_deltam + _deltap - 2.);
    if ((w >= 0.) && (w <= 1.))
    {
        this->scheme_type = 2;
        return coeff; // схема Схема Русанова
    }

    w = sigma * (1. + sigma) * 0.5 + _deltap * sigma * (1. - sigma) * 0.5;
    if ((w >= 0.) && (w <= 1.))
    {
        this->scheme_type = 0;
        return 0.; // схема Лакса-Вендроффа
    }

    w = sigma * (3. - sigma) * 0.5 + _deltam * sigma * (sigma - 1.) * 0.5;
    if ((w >= 0.) && (w <= 1.))
    {
        this->scheme_type = 1;
        return 0.5 * sigma * (sigma - 1.); // схема Бима-Уорминга
    }

    //printf("1");
    this->scheme_type = -1;
    return -1.;
    //return sigma * (sigma * sigma - 1.) / 6.;
}

void Wave3d::_make_step_X()
{
    // Решение уравнения
    // dW/dt + A * dW/dx = 0 

    long int index = 0;
    // Переход к характеристической системе
    // (получим решение в инвариантах Римана)
#pragma omp for private(index)
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
    // -> Обмен приграничными узлами между процессами при решении уравнения dW/dt + A * dW/dx = 0
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    // Обмен данных по оси x. Данные для отправки лежат последовательно.
    // Отправляем данные "соседу" снизу, считываем данные от "соседа" сверху.
    // В случае, если "соседа" снизу нет, то lower_rank = MPI_PROC_NULL,
    // в этом случае вызов функции MPI_Sendrecv завершится без ошибок, операция отправки не осуществится,
    // будет совершенно только считывание.
    // Аналогично, если "сосед" сверху отсутствует, выполнится только отправка данных нижнему процессу.
    MPI_Sendrecv(
        g_rank_curr, 1, lower_subaray_send, lower_rank, 200,
        g_rank_curr, 1, upper_subaray_recv, upper_rank, 200,
        communicator, MPI_STATUS_IGNORE);

    // Аналогично предыдущему вызову функции MPI_Sendrecv
    MPI_Sendrecv(
        g_rank_curr, 1, upper_subaray_send, upper_rank, 201,
        g_rank_curr, 1, lower_subaray_recv, lower_rank, 201,
        communicator, MPI_STATUS_IGNORE);

    //bool pml_2 = false;
    //long int start_pml2;
    //if (starts_f[0] + _size_i - 4 >= I)
    //    pml_2 = true;

    //bool pml_1 = false;
    //long int end_pml1;
    //if (starts_f[0] < 0)
    //    pml_1 = true;

    //bool basic = false;
    //long int start_basic, end_basic;
    //if ((starts_f[0] >= 0) && ((starts_f[0] < I) || (starts_f[0] + _size_i - 3 > I)) || (starts_f[0] < 0) && (starts_f[0] + _size_i - 3 > 0))
    //    basic = true;

    //if (basic)
    //{
    //    if ((starts_f[0] >= 0) && (starts_f[0] + _size_i - 5 <= I))
    //    {
    //        start_basic = starts_f[0];
    //    }
    //}

    //if (pml_1)
    //{
    //    if (starts_f[0] + PML_Size + _size_i - 5 >= PML_Size)
    //        end_pml1 = PML_Size;
    //    else
    //        end_pml1 = starts_f[0] + PML_Size + _size_i - 4;
    //}

    //if (pml_1)
    //{
    //    for (long int j = 2; j < _size_j - 2; ++j)
    //    {
    //        for (long int k = 2; k < _size_k - 2; ++k)
    //        {
    //            // Расчет внутри PML 
    //            for (long int i = starts_f[0] + PML_Size; i < end_pml1; ++i)
    //            {
    //                index = (i + 2 - starts_f[0] - PML_Size) * _I + j * _J + k;
    //                // Демпфирующая функция d(s)
    //                double demp = static_cast<double>(PML_Size - (i - starts_f[0] - PML_Size)) / static_cast<double>(PML_Size) * sigma_max * v_p[index];

    //                // Решение 9-ти независымых уравнений переноса в PML области
    //                this->_PML_solve_system_eq(index, g_rank_curr[index - 2 * _I], g_rank_curr[index - _I], g_rank_curr[index], g_rank_curr[index + _I], g_rank_curr[index + 2 * _I], demp);
    //            }
    //        }
    //    }
    //}

    //if (pml_2)
    //{
    //    if (starts_f[0] + PML_Size < I + PML_Size)
    //        start_pml2 = I + PML_Size;
    //    else
    //        start_pml2 = starts_f[0] + PML_Size;
    //}

    //if (pml_2)
    //{
    //    for (long int j = 2; j < _size_j - 2; ++j)
    //    {
    //        for (long int k = 2; k < _size_k - 2; ++k)
    //        {
    //            // Расчет внутри PML
    //            for (long int i = start_pml2; i < starts_f[0] + PML_Size + _size_i - 4; ++i)
    //            {
    //                index = (i - start_pml2 + 2) * _I + j * _J + k;
    //                // Демпфирующая функция d(s)
    //                double demp = static_cast<double>(i - (I + PML_Size) + 1) / static_cast<double>(PML_Size) * sigma_max * v_p[index];

    //                // Решение 9-ти независымых уравнений переноса в PML области
    //                //this->_PML_transfer_eq_x(i * _I + j * _J + k, demp);
    //                //index = i * _I + j * _J + k;
    //                this->_PML_solve_system_eq(index, g_rank_curr[index - 2 * _I], g_rank_curr[index - _I], g_rank_curr[index], g_rank_curr[index + _I], g_rank_curr[index + 2 * _I], demp);
    //            }
    //        }
    //    }
    //}

    // Решение по монотонным разностным схемам сеточно-характеристическим методом
    long int start_i = 2;
    long int end_i = _size_i - 2;
    // Корректная работа в предположении, что декомпозиция по направлению x проводилась минимум на 2 процесса,
    // т.е область поделилась хотя бы на две под области в данном направлении.
    // Так же, размер подобласти по направлению x больше, чем размер PML - области (обычно PML - 10-15 узлов)
    if (rank_coords[0] == 0)
    {
        start_i = 2 + PML_Size;
        end_i = _size_i - 2;

#pragma omp for private(index)
        for (long int j = 2; j < _size_j - 2; ++j)
        {
            for (long int k = 2; k < _size_k - 2; ++k)
            {
                // Расчет внутри PML 
                for (long int i = 2; i < 2 + PML_Size; ++i)
                {
                    index = i * _I + j * _J + k;
                    // Демпфирующая функция d(s)
                    double demp = static_cast<double>(PML_Size - (i - 2)) / static_cast<double>(PML_Size) * sigma_max * v_p[index];

                    // Решение 9-ти независымых уравнений переноса в PML области
                    //this->_PML_transfer_eq_x(i * _I + j * _J + k, demp);
                    //index = i * _I + j * _J + k;
                    this->_PML_solve_system_eq(index, g_rank_curr[index - 2 * _I], g_rank_curr[index - _I], g_rank_curr[index], g_rank_curr[index + _I], g_rank_curr[index + 2 * _I], demp);
                }
            }
        }
    }

    // Рассчетная область прилегает к границе, а так же содердит PML - область
    if (rank_coords[0] == dims[0] - 1)
    {
        start_i = 2;
        end_i = _size_i - PML_Size - 2;

#pragma omp for private(index)
        for (long int j = 2; j < _size_j - 2; ++j)
        {
            for (long int k = 2; k < _size_k - 2; ++k)
            {
                // Расчет внутри PML
                for (long int i = _size_i - PML_Size - 2; i < _size_i - 2; ++i)
                {
                    index = i * _I + j * _J + k;
                    // Демпфирующая функция d(s)
                    double demp = static_cast<double>((i + 2) - _size_i + PML_Size + 1) / static_cast<double>(PML_Size) * sigma_max * v_p[index];

                    // Решение 9-ти независымых уравнений переноса в PML области
                    //this->_PML_transfer_eq_x(i * _I + j * _J + k, demp);
                    //index = i * _I + j * _J + k;
                    this->_PML_solve_system_eq(index, g_rank_curr[index - 2 * _I], g_rank_curr[index - _I], g_rank_curr[index], g_rank_curr[index + _I], g_rank_curr[index + 2 * _I], demp);
                }
            }
        }
    }

    if (dims[0] == 1)
    {
        start_i = 2 + PML_Size;
        end_i = _size_i - PML_Size - 2;
    }

    // PML область рассчитывается в процессах, чья подобласть примыкает к границе по оси x.
    // Остальные процессы производят рассчет внутри исходной области
#pragma omp for private(index)
    for (long int j = 2; j < _size_j - 2; ++j)
    {
        for (long int k = 2; k < _size_k - 2; ++k)
        {
            // Расчет внутри исходной вычислительной области
            for (long int i = start_i; i < end_i; ++i)
            {
                // Решение 9-ти независымых уравнений переноса
                //this->_transfer_eq_x(i * _I + j * _J + k);
                index = i * _I + j * _J + k;
                this->_solve_system_eq(index, g_rank_curr[index - 2 * _I], g_rank_curr[index - _I], g_rank_curr[index], g_rank_curr[index + _I], g_rank_curr[index + 2 * _I]);
            }
        }
    }
    

    // Обратная замена - переход от характеристической системы к исходной
#pragma omp for private(index)
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


    // Контактные условия - контактный корректор полного слипания
#pragma omp for private(index)
    for (int i = 2; i < _size_i - 2; ++i)
    {
        for (int j = 2; j < _size_j - 2; ++j)
        {
            for (int k = 2; k < _size_k - 2; ++k)
            {
                index = i * _I + j * _J + k;
                if (v_p[index] != v_p[index - _I])
                {
                    double vp_a = v_p[index - _I];
                    double vs_a = v_s[index - _I];
                    double rho_a = Rho[index - _I];
                    Vars w_a = w_rank_curr[index - _I];

                    double vp_b = v_p[index];
                    double vs_b = v_s[index];
                    double rho_b = Rho[index];
                    Vars w_b = w_rank_curr[index];

                    double vs = v_s[index];
                    double vp = v_p[index];
                    double c_3 = v_p[index] * (1. - 2. * vs_b * vs_b / (vp_b * vp_b));

                    double v_x = (rho_a * vp_a * w_a.u_x + rho_b * vp_b * w_b.u_x + w_a.sigma_xx - w_b.sigma_xx
                        - (rho_a * (vp_a - vs_a) + rho_b * (vp_b - vs_b)) / (rho_a * vp_a + rho_b * vp_b)
                        * (rho_a * vp_a * w_a.u_x + rho_b * vp_b * w_b.u_x + w_a.sigma_xx - w_b.sigma_xx)) / (rho_a * vs_a + rho_b * vs_b);

                    double v_y = (rho_a * vs_a * w_a.u_y + rho_b * vs_b * w_b.u_y + w_a.sigma_xy - w_b.sigma_xy) / (rho_a * vs_a + rho_b * vs_b);
                    double v_z = (rho_a * vs_a * w_a.u_z + rho_b * vs_b * w_b.u_z + w_a.sigma_zx - w_b.sigma_zx) / (rho_a * vs_a + rho_b * vs_b);

                    double z[3];
                    z[0] = w_rank_curr[index].u_x - v_x;
                    z[1] = w_rank_curr[index].u_y - v_y;
                    z[2] = w_rank_curr[index].u_z - v_z;

                    w_rank_curr[index].u_x = v_x;
                    w_rank_curr[index].u_y = v_y;
                    w_rank_curr[index].u_z = v_z;


                    //
                    // по формуле граничного корректора с заданной скоростью границы
                    // 
                    // sigma_n+1 = sigma_in + rho * ((z * n) * ((v_p - 2v_s - c_3) * N_00 + c_3 * I) + v_s * (n (*) z + z (*) n))
                    //                     "+" - так как контактная граница слева
                    // (*) - тензорное произведение векторов
                    // n = ( n_0 n_1 n_1 ) - вектор
                    // в направлении x - n = ( 1 0 0 )
                    // z = ( z_0 z_1 z_2 ) - вектор
                    // 
                    // z * n в данном случае = z_0
                    // 
                    //                                           | 1 0 0 |
                    // N_00 = 1/2 * (n0 (*) n0 + n0 (*) n0)   =  | 0 0 0 |
                    //                                           | 0 0 0 |
                    //         | sigma_xx sigma_xy sigma_xz |
                    // sigma = | sigma_xy sigma_yy sigma_yz |
                    //         | sigma_xz sigma_yz sigma_zz |
                    //     
                    //     | 1 0 0 |
                    // I = | 0 1 0 |
                    //     | 0 0 0 |
                    //                     | z_0 z_1 z_2 |   | z_0  0   0 |   | 2*z_0 z_1 z_2 |
                    // n (*) z + z (*) n = |  0   0   0  | + | z_1  0   0 | = |  z_1   0   0  |
                    //                     |  0   0   0  |   | z_2  0   0 |   |  z_2   0   0  |
                    // 
                    // sigma_n+1 - новые значения тензора напряжений после корректировки
                    // sigma_in - значения тензора напряжений расчитанные по разностным схемам до корректировки
                    //
                    // 
                    // | sigma_xx sigma_xy sigma_xz |                                       | 1 0 0 |         | 1 0 0 |           | 2*z_0 z_1 z_2 |
                    // | sigma_xy sigma_yy sigma_yz | += rho * (z_0 * ((v_p - 2v_s - c_3) * | 0 0 0 | + c_3 * | 0 1 0 | ) + v_s * |  z_1   0   0  | )
                    // | sigma_xz sigma_yz sigma_zz |                                       | 0 0 0 |         | 0 0 1 |           |  z_2   0   0  |
                    //
                    //  После раскрытия скобок получаем сследующие формулы
                    // 
                    // sigma_xx += rho * z_0 * v_p
                    // sigma_yy += rho * z_0 * c_3
                    // sigma_zz += rho * z_0 * c_3
                    //
                    // sigma_xy += rho * v_s * z_1
                    // sigma_xz += rho * v_s * z_2
                    // sigma_yz += 0
                    //

                    w_rank_curr[index].sigma_xx += rho_b * z[0] * vp_b;
                    w_rank_curr[index].sigma_yy += rho_b * z[0] * c_3;
                    w_rank_curr[index].sigma_zz += rho_b * z[0] * c_3;

                    w_rank_curr[index].sigma_xy += rho_b * vs_b * z[1];
                    w_rank_curr[index].sigma_zx += rho_b * vs_b * z[2];
                    //w_rank_curr[index].sigma_yz += 0.;
                }

                if (v_p[index] != v_p[index + _I])
                {
                    double vp_a = v_p[index];
                    double vs_a = v_s[index];
                    double rho_a = Rho[index];
                    Vars w_a = w_rank_curr[index];

                    double vp_b = v_p[index + _I];
                    double vs_b = v_s[index + _I];
                    double rho_b = Rho[index + _I];
                    Vars w_b = w_rank_curr[index + _I];

                    double vs = v_s[index];
                    double vp = v_p[index];
                    double c_3 = v_p[index] * (1. - 2. * vs_a * vs_a / (vp_a * vp_a));

                    double v_x = (rho_a * vp_a * w_a.u_x + rho_b * vp_b * w_b.u_x + w_b.sigma_xx - w_a.sigma_xx
                        - (rho_a * (vp_a - vs_a) + rho_b * (vp_b - vs_b)) / (rho_a * vp_a + rho_b * vp_b)
                        * (rho_a * vp_a * w_a.u_x + rho_b * vp_b * w_b.u_x + w_b.sigma_xx - w_a.sigma_xx)) / (rho_a * vs_a + rho_b * vs_b);

                    double v_y = (rho_a * vs_a * w_a.u_y + rho_b * vs_b * w_b.u_y + w_b.sigma_xy - w_a.sigma_xy) / (rho_a * vs_a + rho_b * vs_b);
                    double v_z = (rho_a * vs_a * w_a.u_z + rho_b * vs_b * w_b.u_z + w_b.sigma_zx - w_a.sigma_zx) / (rho_a * vs_a + rho_b * vs_b);

                    double z[3];
                    z[0] = w_rank_curr[index].u_x - v_x;
                    z[1] = w_rank_curr[index].u_y - v_y;
                    z[2] = w_rank_curr[index].u_z - v_z;

                    w_rank_curr[index].u_x = v_x;
                    w_rank_curr[index].u_y = v_y;
                    w_rank_curr[index].u_z = v_z;


                    //
                    // по формуле граничного корректора с заданной скоростью границы
                    // 
                    // sigma_n+1 = sigma_in - rho * ((z * n) * ((v_p - 2v_s - c_3) * N_00 + c_3 * I) + v_s * (n (*) z + z (*) n))
                    //                     "-" - так как контактная граница справа
                    // (*) - тензорное произведение векторов
                    // n = ( n_0 n_1 n_1 ) - вектор
                    // в направлении x - n = ( 1 0 0 )
                    // z = ( z_0 z_1 z_2 ) - вектор
                    // 
                    // z * n в данном случае = z_0
                    // 
                    //                                           | 1 0 0 |
                    // N_00 = 1/2 * (n0 (*) n0 + n0 (*) n0)   =  | 0 0 0 |
                    //                                           | 0 0 0 |
                    //         | sigma_xx sigma_xy sigma_xz |
                    // sigma = | sigma_xy sigma_yy sigma_yz |
                    //         | sigma_xz sigma_yz sigma_zz |
                    //     
                    //     | 1 0 0 |
                    // I = | 0 1 0 |
                    //     | 0 0 0 |
                    //                     | z_0 z_1 z_2 |   | z_0  0   0 |   | 2*z_0 z_1 z_2 |
                    // n (*) z + z (*) n = |  0   0   0  | + | z_1  0   0 | = |  z_1   0   0  |
                    //                     |  0   0   0  |   | z_2  0   0 |   |  z_2   0   0  |
                    // 
                    // sigma_n+1 - новые значения тензора напряжений после корректировки
                    // sigma_in - значения тензора напряжений расчитанные по разностным схемам до корректировки
                    //
                    // 
                    // | sigma_xx sigma_xy sigma_xz |                                       | 1 0 0 |         | 1 0 0 |           | 2*z_0 z_1 z_2 |
                    // | sigma_xy sigma_yy sigma_yz | -= rho * (z_0 * ((v_p - 2v_s - c_3) * | 0 0 0 | + c_3 * | 0 1 0 | ) + v_s * |  z_1   0   0  | )
                    // | sigma_xz sigma_yz sigma_zz |                                       | 0 0 0 |         | 0 0 1 |           |  z_2   0   0  |
                    //
                    //  После раскрытия скобок получаем сследующие формулы
                    // 
                    // sigma_xx -= rho * z_0 * v_p
                    // sigma_yy -= rho * z_0 * c_3
                    // sigma_zz -= rho * z_0 * c_3
                    //
                    // sigma_xy -= rho * v_s * z_1
                    // sigma_xz -= rho * v_s * z_2
                    // sigma_yz -= 0
                    //

                    w_rank_curr[index].sigma_xx -= rho_b * z[0] * vp_b;
                    w_rank_curr[index].sigma_yy -= rho_b * z[0] * c_3;
                    w_rank_curr[index].sigma_zz -= rho_b * z[0] * c_3;

                    w_rank_curr[index].sigma_xy -= rho_b * vs_b * z[1];
                    w_rank_curr[index].sigma_zx -= rho_b * vs_b * z[2];
                    //w_rank_curr[index].sigma_yz += 0.;
                }
            }
        }
    }


    Vars w_0, w_m, w_p1, w_p2;
    // Поглащающие граничные условия по направлению x. Расчет с помощью мнимых точек на внешей границе PML области.
    // Для границы (i = 2)
    // Для данной подобласти (rank_coords[0] == 0), "соседа" сверху нет,
    // в область памяти, где должны были лежать границные узлы от соседа,
    // записываются данные граничные условия
    if (rank_coords[0] == 0)
    {
#pragma omp for private(index, w_0, w_m, w_p1, w_p2)
        for (long int j = 2; j < _size_j - 2; ++j)
        {
            for (long int k = 2; k < _size_k - 2; ++k)
            {
                // i = 2
                index = 2 * _I + j * _J + k;

                // Для первой мнимой точки
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

                // Для второй мнимой точки
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


    // Поглащающие граничные условия по направлению x. Расчет с помощью мнимых точек на внешей границе PML области.
    // Для границы (i = _size_i - 3)
    // Для данной подобласти (rank_coords[0] == dims[0] - 1), "соседа" справа нет,
    // в область памяти, где должны были лежать границные узлы от соседа,
    // записываются данные граничные условия
    if (rank_coords[0] == dims[0] - 1)
    {
#pragma omp for private(index, w_0, w_m, w_p1, w_p2)
        for (long int j = 2; j < _size_j - 2; ++j)
        {
            for (long int k = 2; k < _size_k - 2; ++k)
            {
                // i = _size_i - 3
                index = (_size_i - 3) * _I + j * _J + k;

                // Для первой мнимой точки
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

                // Для второй мнимой точки
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
    // Решение уравнения
    // dW/dt + A * dW/dy = 0 

    long int index = 0;
    // Переход к характеристической системе
    // (получим решение в инвариантах Римана)
#pragma omp for private(index)
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
    // -> Обмен приграничными узлами между процессами при решении уравнения dW/dt + A * dW/dy = 0
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // обьмен данных между потоками
    // Отправляем данные "соседу" справа, считываем данные от "соседа" слева.
    // В случае, если "соседа" справа нет, то right_rank = MPI_PROC_NULL,
    // в этом случае вызов функции MPI_Sendrecv завершится без ошибок, операция отправки не осуществится,
    // будет совершенно только считывание.
    // Аналогично, если "сосед" слева отсутствует, выполнится только отправка данных соседнему процессу.
    MPI_Sendrecv(
        g_rank_curr, 1, right_subaray_send, right_rank, 300,
        g_rank_curr, 1, left_subaray_recv, left_rank, 300,
        communicator, MPI_STATUS_IGNORE);

    // Аналогично предыдущему вызову функции MPI_Sendrecv
    MPI_Sendrecv(
        g_rank_curr, 1, left_subaray_send, left_rank, 301,
        g_rank_curr, 1, right_subaray_recv, right_rank, 301,
        communicator, MPI_STATUS_IGNORE);

    // Решение по монотонным разностным схемам сеточно-характеристическим методом
    long int start_j = 2;
    long int end_j = _size_j - 2;
    // Корректная работа в предположении, что декомпозиция по направлению y проводилась минимум на 2 процесса,
    // т.е область поделилась хотя бы на две под области в данном направлении.
    // Так же, размер подобласти по направлению y больше, чем размер PML - области (обычно PML - 10-15 узлов)
    if (rank_coords[1] == 0)
    {
        start_j = 2 + PML_Size;
        end_j = _size_j - 2;

#pragma omp for private(index)
        for (long int i = 2; i < _size_i - 2; ++i)
        {
            for (long int k = 2; k < _size_k - 2; ++k)
            {
                // Расчет внутри PML 
                for (long int j = 2; j < 2 + PML_Size; ++j)
                {
                    index = i * _I + j * _J + k;
                    // Демпфирующая функция d(s)
                    double demp = static_cast<double>(PML_Size - (j - 2)) / static_cast<double>(PML_Size) * sigma_max * v_p[index];

                    // Решение 9-ти независымых уравнений переноса в PML области
                    //this->_PML_transfer_eq_y(i * _I + j * _J + k, demp);
                    //index = i * _I + j * _J + k;
                    this->_PML_solve_system_eq(index, g_rank_curr[index - 2 * _J], g_rank_curr[index - _J], g_rank_curr[index], g_rank_curr[index + _J], g_rank_curr[index + 2 * _J], demp);
                }
            }
        }
    }

    // Рассчетная область прилегает к границе, а так же содердит PML - область
    if (rank_coords[1] == dims[1] - 1)
    {
        start_j = 2;
        end_j = _size_j - PML_Size - 2;

#pragma omp for private(index)
        for (long int i = 2; i < _size_i - 2; ++i)
        {
            for (long int k = 2; k < _size_k - 2; ++k)
            {
                // Расчет внутри PML
                for (long int j = _size_j - PML_Size - 2; j < _size_j - 2; ++j)
                {
                    index = i * _I + j * _J + k;
                    // Демпфирующая функция d(s)
                    double demp = static_cast<double>((j + 2) - _size_j + PML_Size + 1) / static_cast<double>(PML_Size) * sigma_max * v_p[index];

                    // Решение 9-ти независымых уравнений переноса в PML области
                    //this->_PML_transfer_eq_y(i * _I + j * _J + k, demp);
                    //index = i * _I + j * _J + k;
                    this->_PML_solve_system_eq(index, g_rank_curr[index - 2 * _J], g_rank_curr[index - _J], g_rank_curr[index], g_rank_curr[index + _J], g_rank_curr[index + 2 * _J], demp);
                }
            }
        }
    }

    if (dims[1] == 1)
    {
        start_j = 2 + PML_Size;
        end_j = _size_i - PML_Size - 2;
    }

    // PML область рассчитывается в процессах, чья подобласть примыкает к границе по оси j.
    // Остальные процессы производят рассчет внутри исходной области
#pragma omp for private(index)
    for (long int i = 2; i < _size_i - 2; ++i)
    {
        for (long int k = 2; k < _size_k - 2; ++k)
        {
            // Расчет внутри исходной вычислительной области
            for (long int j = start_j; j < end_j; ++j)
            {
                // Решение 9-ти независымых уравнений переноса
                //this->_transfer_eq_y(i * _I + j * _J + k);
                index = i * _I + j * _J + k;
                this->_solve_system_eq(index, g_rank_curr[index - 2 * _J], g_rank_curr[index - _J], g_rank_curr[index], g_rank_curr[index + _J], g_rank_curr[index + 2 * _J]);
            }
        }
    }


    // обратная замена - переход от характеристической системы к исходной
#pragma omp for private(index)
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


    // Контактные условия - контактный корректор полного слипания
#pragma omp for private(index)
    for (int i = 2; i < _size_i - 2; ++i)
    {
        for (int j = 2; j < _size_j - 2; ++j)
        {
            for (int k = 2; k < _size_k - 2; ++k)
            {
                index = i * _I + j * _J + k;
                if (v_p[index] != v_p[index - _J])
                {
                    double vp_a = v_p[index - _J];
                    double vs_a = v_s[index - _J];
                    double rho_a = Rho[index - _J];
                    Vars w_a = w_rank_curr[index - _J];

                    double vp_b = v_p[index];
                    double vs_b = v_s[index];
                    double rho_b = Rho[index];
                    Vars w_b = w_rank_curr[index];

                    double vs = v_s[index];
                    double vp = v_p[index];
                    double c_3 = v_p[index] * (1. - 2. * vs_b * vs_b / (vp_b * vp_b));

                    double v_x = (rho_a * vs_a * w_a.u_x + rho_b * vs_b * w_b.u_x + w_a.sigma_xy - w_b.sigma_xy) / (rho_a * vs_a + rho_b * vs_b);

                    double v_y = (rho_a * vp_a * w_a.u_y + rho_b * vp_b * w_b.u_y + w_a.sigma_yy - w_b.sigma_yy
                        - (rho_a * (vp_a - vs_a) + rho_b * (vp_b - vs_b)) / (rho_a * vp_a + rho_b * vp_b)
                        * (rho_a * vp_a * w_a.u_y + rho_b * vp_b * w_b.u_y + w_a.sigma_yy - w_b.sigma_yy)) / (rho_a * vs_a + rho_b * vs_b);

                    double v_z = (rho_a * vs_a * w_a.u_z + rho_b * vs_b * w_b.u_z + w_a.sigma_yz - w_b.sigma_yz) / (rho_a * vs_a + rho_b * vs_b);

                    double z[3];
                    z[0] = w_b.u_x - v_x;
                    z[1] = w_b.u_y - v_y;
                    z[2] = w_b.u_z - v_z;

                    w_b.u_x = v_x;
                    w_b.u_y = v_y;
                    w_b.u_z = v_z;


                    //
                    // по формуле граничного корректора с заданной скоростью границы
                    // 
                    // sigma_n+1 = sigma_in + rho * ((z * n) * ((v_p - 2v_s - c_3) * N_11 + c_3 * I) + v_s * (n (*) z + z (*) n))
                    //                     "+" - так как контактная граница слева
                    // (*) - тензорное произведение векторов
                    // n = ( n_0 n_1 n_1 ) - вектор
                    // в направлении y - n = ( 0 1 0 )
                    // z = ( z_0 z_1 z_2 ) - вектор
                    // 
                    // z * n в данном случае = z_1
                    // 
                    //                                           | 0 0 0 |
                    // N_11 = 1/2 * (n1 (*) n1 + n1 (*) n1)   =  | 0 1 0 |
                    //                                           | 0 0 0 |
                    //         | sigma_xx sigma_xy sigma_xz |
                    // sigma = | sigma_xy sigma_yy sigma_yz |
                    //         | sigma_xz sigma_yz sigma_zz |
                    //     
                    //     | 1 0 0 |
                    // I = | 0 1 0 |
                    //     | 0 0 1 |
                    //                     |  0   0   0  |   |  0  z_0  0 |   |  0    z_0    0  |
                    // n (*) z + z (*) n = | z_0 z_1 z_2 | + |  0  z_1  0 | = | z_0  2*z_1  z_2 |
                    //                     |  0   0   0  |   |  0  z_2  0 |   |  0    z_2    0  |
                    // 
                    // sigma_n+1 - новые значения тензора напряжений после корректировки
                    // sigma_in - значения тензора напряжений расчитанные по разностным схемам до корректировки
                    //
                    // 
                    // | sigma_xx sigma_xy sigma_xz |                                       | 0 0 0 |         | 1 0 0 |           |  0    z_0    0  |
                    // | sigma_xy sigma_yy sigma_yz | += rho * (z_1 * ((v_p - 2v_s - c_3) * | 0 1 0 | + c_3 * | 0 1 0 | ) + v_s * | z_0  2*z_1  z_2 | )
                    // | sigma_xz sigma_yz sigma_zz |                                       | 0 0 0 |         | 0 0 1 |           |  0    z_2    0  |
                    //
                    //  После раскрытия скобок получаем сследующие формулы
                    // 
                    // sigma_xx += rho * z_1 * c_3
                    // sigma_yy += rho * z_1 * v_p
                    // sigma_zz += rho * z_1 * c_3
                    //
                    // sigma_xy += rho * v_s * z_0
                    // sigma_xz += 0
                    // sigma_yz += rho * v_s * z_2
                    //

                    w_b.sigma_xx += rho_b * z[1] * c_3;
                    w_b.sigma_yy += rho_b * z[1] * vp_b;
                    w_b.sigma_zz += rho_b * z[1] * c_3;

                    w_b.sigma_xy += rho_b * vs_b * z[0];
                    //w_b.sigma_zx += 0.;
                    w_b.sigma_yz += rho_b * vs_b * z[2];

                    w_rank_curr[index] = w_b;
                }

                if (v_p[index] != v_p[index + _J])
                {
                    double vp_a = v_p[index];
                    double vs_a = v_s[index];
                    double rho_a = Rho[index];
                    Vars w_a = w_rank_curr[index];

                    double vp_b = v_p[index + _J];
                    double vs_b = v_s[index + _J];
                    double rho_b = Rho[index + _J];
                    Vars w_b = w_rank_curr[index + _J];

                    double vs = v_s[index];
                    double vp = v_p[index];
                    double c_3 = v_p[index] * (1. - 2. * vs_a * vs_a / (vp_a * vp_a));

                    double v_x = (rho_a * vs_a * w_a.u_x + rho_b * vs_b * w_b.u_x + w_b.sigma_xy - w_a.sigma_xy) / (rho_a * vs_a + rho_b * vs_b);

                    double v_y = (rho_a * vp_a * w_a.u_y + rho_b * vp_b * w_b.u_y + w_b.sigma_yy - w_a.sigma_yy
                        - (rho_a * (vp_a - vs_a) + rho_b * (vp_b - vs_b)) / (rho_a * vp_a + rho_b * vp_b)
                        * (rho_a * vp_a * w_a.u_y + rho_b * vp_b * w_b.u_y + w_b.sigma_yy - w_a.sigma_yy)) / (rho_a * vs_a + rho_b * vs_b);

                    double v_z = (rho_a * vs_a * w_a.u_z + rho_b * vs_b * w_b.u_z + w_b.sigma_yz - w_a.sigma_yz) / (rho_a * vs_a + rho_b * vs_b);

                    double z[3];
                    z[0] = w_a.u_x - v_x;
                    z[1] = w_a.u_y - v_y;
                    z[2] = w_a.u_z - v_z;

                    w_a.u_x = v_x;
                    w_a.u_y = v_y;
                    w_a.u_z = v_z;


                    //
                    // по формуле граничного корректора с заданной скоростью границы
                    // 
                    // sigma_n+1 = sigma_in - rho * ((z * n) * ((v_p - 2v_s - c_3) * N_11 + c_3 * I) + v_s * (n (*) z + z (*) n))
                    //                     "-" - так как контактная граница справа
                    // (*) - тензорное произведение векторов
                    // n = ( n_0 n_1 n_1 ) - вектор
                    // в направлении x - n = ( 0 1 0 )
                    // z = ( z_0 z_1 z_2 ) - вектор
                    // 
                    // z * n в данном случае = z_1
                    // 
                    //                                           | 0 0 0 |
                    // N_11 = 1/2 * (n1 (*) n1 + n1 (*) n1)   =  | 0 1 0 |
                    //                                           | 0 0 0 |
                    //         | sigma_xx sigma_xy sigma_xz |
                    // sigma = | sigma_xy sigma_yy sigma_yz |
                    //         | sigma_xz sigma_yz sigma_zz |
                    //     
                    //     | 1 0 0 |
                    // I = | 0 1 0 |
                    //     | 0 0 1 |
                    //                     |  0   0   0  |   |  0  z_0  0 |   |  0    z_0    0  |
                    // n (*) z + z (*) n = | z_0 z_1 z_2 | + |  0  z_1  0 | = | z_0  2*z_1  z_2 |
                    //                     |  0   0   0  |   |  0  z_2  0 |   |  0    z_2    0  |
                    // 
                    // sigma_n+1 - новые значения тензора напряжений после корректировки
                    // sigma_in - значения тензора напряжений расчитанные по разностным схемам до корректировки
                    //
                    // 
                    // | sigma_xx sigma_xy sigma_xz |                                       | 0 0 0 |         | 1 0 0 |           |  0    z_0    0  |
                    // | sigma_xy sigma_yy sigma_yz | -= rho * (z_1 * ((v_p - 2v_s - c_3) * | 0 1 0 | + c_3 * | 0 1 0 | ) + v_s * | z_0  2*z_1  z_2 | )
                    // | sigma_xz sigma_yz sigma_zz |                                       | 0 0 0 |         | 0 0 1 |           |  0    z_2    0  |
                    //
                    //  После раскрытия скобок получаем сследующие формулы
                    // 
                    // sigma_xx -= rho * z_1 * c_3
                    // sigma_yy -= rho * z_1 * v_p
                    // sigma_zz -= rho * z_1 * c_3
                    //
                    // sigma_xy -= rho * v_s * z_0
                    // sigma_xz -= 0
                    // sigma_yz -= rho * v_s * z_2
                    //

                    w_a.sigma_xx -= rho_b * z[1] * c_3;
                    w_a.sigma_yy -= rho_b * z[1] * vp_b;
                    w_a.sigma_zz -= rho_b * z[1] * c_3;

                    w_a.sigma_xy -= rho_b * vs_b * z[0];
                    w_a.sigma_yz -= rho_b * vs_b * z[2];

                    w_rank_curr[index] = w_a;
                }
            }
        }
    }


    Vars w_0, w_m, w_p1, w_p2;
    // Поглащающие граничные условия по направлению y. Расчет с помощью мнимых точек на внешей границе PML области.
    // Для границы слева (j = 2)
    // j = 0, 1 - дополнительные мнимые точки
    //
    // Для данной подобласти (rank_coords[1] == 0), "соседа" слева нет,
    // в область памяти, где должны были лежать границные узлы от соседа,
    // записываются данные граничные условия
    if (rank_coords[1] == 0)
    {
#pragma omp for private(index, w_0, w_m, w_p1, w_p2)
        for (long int i = 2; i < _size_i - 2; ++i)
        {
            for (long int k = 2; k < _size_k - 2; ++k)
            {
                // j = 2
                index = i * _I + 2 * _J + k;

                // Для первой мнимой точки
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

                // Для второй мнимой точки
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


    // Поглащающие граничные условия по направлению y. Расчет с помощью мнимых точек на внешей границе PML области.
    // Для границы слева (j = _size_j - 3)
    // j = _size_j - 2, _size_j - 1 - дополнительные мнимые точки
    //
    // Для данной подобласти (rank_coords[1] == dims[1] - 1), "соседа" справа нет,
    // в область памяти, где должны были лежать границные узлы от соседа,
    // записываются данные граничные условия
    if (rank_coords[1] == dims[1] - 1)
    {
#pragma omp for private(index, w_0, w_m, w_p1, w_p2)
        for (long int i = 2; i < _size_i - 2; ++i)
        {
            for (long int k = 2; k < _size_k - 2; ++k)
            {
                // j = _size_j - 3
                index = i * _I + (_size_j - 3) * _J + k;

                // Для первой мнимой точки
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

                // Для второй мнимой точки
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
    // Решение уравнения
    // dW/dt + A * dW/dz = 0 

    long int index = 0;
    Vars w_0, w_m, w_p1, w_p2;
    // Поглащающие граничные условия по направлению z. Расчет с помощью мнимых точек на внешей границе PML области
    // Для границы слева
    // k = 2
    // к = 0, 1 - дополнительные мнимые точки
#pragma omp for private(index, w_0, w_m, w_p1, w_p2)
    for (long int i = 2; i < _size_i - 2; ++i)
    {
        for (long int j = 2; j < _size_j - 2; ++j)
        {
            // k = 2
            index = i * _I + j * _J + 2;

            // Для первой мнимой точки
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

            // Для второй мнимой точки
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


    // Для правой границы
    // k = _size_k - 3
    // k = _size_k - 2, _size_k - 1 - дополнительные мнимые точки
#pragma omp for private(index, w_0, w_m, w_p1, w_p2)
    for (long int i = 2; i < _size_i - 2; ++i)
    {
        for (long int j = 2; j < _size_j - 2; ++j)
        {
            index = i * _I + j * _J + _size_k - 3;

            // для первой мнимой точки
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

            // для второй мнимой точки
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


    // Граничные условия - отражение от границы на поверхности z = 0:
    // sigma_zz = sigma_zy = sigma_zx = 0
    // Для переменных u_x, u_y, u_z, sigma_xx, sigma_xy, sigma_yy граничные условия не заданы! => PML
    // k = 2 + PML_Size
#pragma omp for private(index)
    for (long int i = 2; i < _size_i - 2; ++i)
    {
        for (long int j = 2; j < _size_j - 2; ++j)
        {
            index = i * _I + j * _J + (2 + PML_Size);
            w_rank_curr[index].sigma_zx = 0.;
            w_rank_curr[index].sigma_yz = 0.;
            w_rank_curr[index].sigma_zz = 0.;

            index--;
            w_rank_curr[index].sigma_zx = 0.;
            w_rank_curr[index].sigma_yz = 0.;
            w_rank_curr[index].sigma_zz = 0.;
        }
    }


    // Переход к характеристической системе
#pragma omp for private(index)
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


    // Решение по монотонным разностным схемам сеточно-характеристическим методом
#pragma omp for private(index)
    for (long int i = 2; i < _size_i - 2; ++i)
    {
        for (long int j = 2; j < _size_j - 2; ++j)
        {
            // Расчет внутри PML
            for (long int k = 2; k < 2 + PML_Size; ++k)
            {
                index = i * _I + j * _J + k;
                // Демпфирующая функция d(s)
                double demp = static_cast<double>(PML_Size - (k - 2)) / static_cast<double>(PML_Size) * sigma_max * v_p[index];

                // Решение 9-ти независымых уравнений переноса в PML области
                //this->_PML_transfer_eq_z(i * _I + j * _J + k, demp);
                //index = i * _I + j * _J + k;
                this->_PML_solve_system_eq(index, g_rank_curr[index - 2], g_rank_curr[index - 1], g_rank_curr[index], g_rank_curr[index + 1], g_rank_curr[index + 2], demp);
            }


            // Расчет внутри исходной вычислительной области
            for (long int k = 2 + PML_Size; k < 2 + I + PML_Size; ++k)
            {
                // Решение 9-ти независымых уравнений переноса
                //this->_transfer_eq_z(i * _I + j * _J + k);
                index = i * _I + j * _J + k;
                this->_solve_system_eq(index, g_rank_curr[index - 2], g_rank_curr[index - 1], g_rank_curr[index], g_rank_curr[index + 1], g_rank_curr[index + 2]);
            }


            // Расчет внутри PML
            for (long int k = 2 + I + PML_Size; k < 2 + I + 2 * PML_Size; ++k)
            {
                index = i * _I + j * _J + k;
                // Демпфирующая функция d(s)
                double demp = static_cast<double>((k - 2) - I - PML_Size + 1) / static_cast<double>(PML_Size) * sigma_max * v_p[index];

                // Решение 9-ти независымых уравнений переноса в PML области
                //this->_PML_transfer_eq_z(i * _I + j * _J + k, demp);
                //index = i * _I + j * _J + k;
                this->_PML_solve_system_eq(index, g_rank_curr[index - 2], g_rank_curr[index - 1], g_rank_curr[index], g_rank_curr[index + 1], g_rank_curr[index + 2], demp);
            }
        }
    }


    // обратная замена - переход от характеристической системы к исходной
#pragma omp for private(index)
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


    // Контактные условия - контактный корректор полного слипания
#pragma omp for private(index)
    for (int i = 2; i < _size_i - 2; ++i)
    {
        for (int j = 2; j < _size_j - 2; ++j)
        {
            for (int k = 3; k < _size_k - 3; ++k)
            {
                index = i * _I + j * _J + k;
                if (v_p[index] != v_p[index - 1])
                {
                    double vp_a = v_p[index];
                    double vs_a = v_s[index];
                    double rho_a = Rho[index];
                    Vars w_a = w_rank_curr[index];

                    double vp_b = v_p[index - 1];
                    double vs_b = v_s[index - 1];
                    double rho_b = Rho[index - 1];
                    Vars w_b = w_rank_curr[index - 1];

                    double c_3 = vp_a - 2. * vs_a * vs_a / vp_a;

                    double v_x = (rho_a * vs_a * w_a.u_x + rho_b * vs_b * w_b.u_x + w_a.sigma_zx - w_b.sigma_zx) / (rho_a * vs_a + rho_b * vs_b);
                    double v_y = (rho_a * vs_a * w_a.u_y + rho_b * vs_b * w_b.u_y + w_a.sigma_yz - w_b.sigma_yz) / (rho_a * vs_a + rho_b * vs_b);

                    double v_z = (rho_a * vp_a * w_a.u_z + rho_b * vp_b * w_b.u_z + w_a.sigma_zz - w_b.sigma_zz
                        - (rho_a * (vp_a - vs_a) + rho_b * (vp_b - vs_b)) / (rho_a * vp_a + rho_b * vp_b)
                        * (rho_a * vp_a * w_a.u_z + rho_b * vp_b * w_b.u_z + w_a.sigma_zz - w_b.sigma_zz)) / (rho_a * vs_a + rho_b * vs_b);

                    double z[3];
                    z[0] = w_rank_curr[index].u_x - v_x;
                    z[1] = w_rank_curr[index].u_y - v_y;
                    z[2] = w_rank_curr[index].u_z - v_z;

                    w_rank_curr[index].u_x = v_x;
                    w_rank_curr[index].u_y = v_y;
                    w_rank_curr[index].u_z = v_z;

                    //
                    // по формуле граничного корректора с заданной скоростью границы
                    // 
                    // sigma_n+1 = sigma_in + rho * ((z * n) * ((v_p - 2v_s - c_3) * N_22 + c_3 * I) + v_s * (n (*) z + z (*) n))
                    //                     "+" - так как контактная граница слева
                    // (*) - тензорное произведение векторов
                    // n = ( n_0 n_1 n_1 ) - вектор
                    // в направлении z - n = ( 0 0 1 )
                    // z = ( z_0 z_1 z_2 ) - вектор
                    // 
                    // z * n в данном случае = z_2
                    // 
                    //                                           | 0 0 0 |
                    // N_22 = 1/2 * (n2 (*) n2 + n2 (*) n2)   =  | 0 0 0 |
                    //                                           | 0 0 1 |
                    //         | sigma_xx sigma_xy sigma_xz |
                    // sigma = | sigma_xy sigma_yy sigma_yz |
                    //         | sigma_xz sigma_yz sigma_zz |
                    //     
                    //     | 1 0 0 |
                    // I = | 0 1 0 |
                    //     | 0 0 1 |
                    //                     |  0   0   0  |   |  0  0  z_0 |   |  0    0    z_0  |
                    // n (*) z + z (*) n = |  0   0   0  | + |  0  0  z_1 | = |  0    0    z_1  |
                    //                     | z_0 z_1 z_2 |   |  0  0  z_2 |   | z_0  z_1  2*z_2 |
                    // 
                    // sigma_n+1 - новые значения тензора напряжений после корректировки
                    // sigma_in - значения тензора напряжений расчитанные по разностным схемам до корректировки
                    //
                    // 
                    // | sigma_xx sigma_xy sigma_xz |                                       | 0 0 0 |         | 1 0 0 |           |  0    0    z_0  |
                    // | sigma_xy sigma_yy sigma_yz | += rho * (z_2 * ((v_p - 2v_s - c_3) * | 0 0 0 | + c_3 * | 0 1 0 | ) + v_s * |  0    0    z_1  | )
                    // | sigma_xz sigma_yz sigma_zz |                                       | 0 0 1 |         | 0 0 1 |           | z_0  z_1  2*z_2 |
                    //
                    //  После раскрытия скобок получаем сследующие формулы
                    // 
                    // sigma_xx += rho * z_2 * c_3
                    // sigma_yy += rho * z_2 * c_3
                    // sigma_zz += rho * z_2 * v_p
                    //
                    // sigma_xy += 0
                    // sigma_xz += rho * v_s * z_0
                    // sigma_yz += rho * v_s * z_1
                    //

                    w_rank_curr[index].sigma_xx += rho_a * z[2] * c_3;
                    w_rank_curr[index].sigma_yy += rho_a * z[2] * c_3;
                    w_rank_curr[index].sigma_zz += rho_a * z[2] * vp_a;

                    w_rank_curr[index].sigma_zx += rho_a * vs_a * z[0];
                    w_rank_curr[index].sigma_yz += rho_a * vs_a * z[1];
                }

                if (v_p[index] != v_p[index + 1])
                {
                    double vp_a = v_p[index];
                    double vs_a = v_s[index];
                    double rho_a = Rho[index];
                    Vars w_a = w_rank_curr[index];

                    double vp_b = v_p[index + 1];
                    double vs_b = v_s[index + 1];
                    double rho_b = Rho[index + 1];
                    Vars w_b = w_rank_curr[index + 1];

                    double c_3 = vp_a - 2. * vs_a * vs_a / vp_a;

                    double v_x = (rho_a * vs_a * w_a.u_x + rho_b * vs_b * w_b.u_x + w_b.sigma_zx - w_a.sigma_zx) / (rho_a * vs_a + rho_b * vs_b);
                    double v_y = (rho_a * vs_a * w_a.u_y + rho_b * vs_b * w_b.u_y + w_b.sigma_yz - w_a.sigma_yz) / (rho_a * vs_a + rho_b * vs_b);

                    double v_z = (rho_a * vp_a * w_a.u_z + rho_b * vp_b * w_b.u_z + w_b.sigma_zz - w_a.sigma_zz
                        - (rho_a * (vp_a - vs_a) + rho_b * (vp_b - vs_b)) / (rho_a * vp_a + rho_b * vp_b)
                        * (rho_a * vp_a * w_a.u_z + rho_b * vp_b * w_b.u_z + w_b.sigma_zz - w_a.sigma_zz)) / (rho_a * vs_a + rho_b * vs_b);

                    double z[3];
                    z[0] = w_rank_curr[index].u_x - v_x;
                    z[1] = w_rank_curr[index].u_y - v_y;
                    z[2] = w_rank_curr[index].u_z - v_z;

                    w_rank_curr[index].u_x = v_x;
                    w_rank_curr[index].u_y = v_y;
                    w_rank_curr[index].u_z = v_z;

                    //
                    // по формуле граничного корректора с заданной скоростью границы
                    // 
                    // sigma_n+1 = sigma_in - rho * ((z * n) * ((v_p - 2v_s - c_3) * N_22 + c_3 * I) + v_s * (n (*) z + z (*) n))
                    //                     "-" - так как контактная граница справа
                    // (*) - тензорное произведение векторов
                    // n = ( n_0 n_1 n_1 ) - вектор
                    // в направлении x - n = ( 0 0 1 )
                    // z = ( z_0 z_1 z_2 ) - вектор
                    // 
                    // z * n в данном случае = z_2
                    // 
                    //                                           | 0 0 0 |
                    // N_22 = 1/2 * (n2 (*) n2 + n2 (*) n2)   =  | 0 0 0 |
                    //                                           | 0 0 1 |
                    //         | sigma_xx sigma_xy sigma_xz |
                    // sigma = | sigma_xy sigma_yy sigma_yz |
                    //         | sigma_xz sigma_yz sigma_zz |
                    //     
                    //     | 1 0 0 |
                    // I = | 0 1 0 |
                    //     | 0 0 1 |
                    //                     |  0   0   0  |   |  0  0  z_0 |   |  0    0    z_0  |
                    // n (*) z + z (*) n = |  0   0   0  | + |  0  0  z_1 | = |  0    0    z_1  |
                    //                     | z_0 z_1 z_2 |   |  0  0  z_2 |   | z_0  z_1  2*z_2 |
                    // 
                    // sigma_n+1 - новые значения тензора напряжений после корректировки
                    // sigma_in - значения тензора напряжений расчитанные по разностным схемам до корректировки
                    //
                    // 
                    // | sigma_xx sigma_xy sigma_xz |                                       | 0 0 0 |         | 1 0 0 |           |  0    0    z_0  |
                    // | sigma_xy sigma_yy sigma_yz | -= rho * (z_2 * ((v_p - 2v_s - c_3) * | 0 0 0 | + c_3 * | 0 1 0 | ) + v_s * |  0    0    z_1  | )
                    // | sigma_xz sigma_yz sigma_zz |                                       | 0 0 1 |         | 0 0 1 |           | z_0  z_1  2*z_2 |
                    //
                    //  После раскрытия скобок получаем сследующие формулы
                    // 
                    // sigma_xx -= rho * z_2 * c_3
                    // sigma_yy -= rho * z_2 * c_3
                    // sigma_zz -= rho * z_2 * v_p
                    //
                    // sigma_xy -= 0
                    // sigma_xz -= rho * v_s * z_0
                    // sigma_yz -= rho * v_s * z_1
                    //

                    w_rank_curr[index].sigma_xx -= rho_a * z[2] * c_3;
                    w_rank_curr[index].sigma_yy -= rho_a * z[2] * c_3;
                    w_rank_curr[index].sigma_zz -= rho_a * z[2] * vp_a;

                    w_rank_curr[index].sigma_zx -= rho_a * vs_a * z[0];
                    w_rank_curr[index].sigma_yz -= rho_a * vs_a * z[1];
                }
            }
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
                v = w_curr[i * _I + j * _J + k];

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
                v = w_curr[i * _I + j * _J + k];

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
                v = w_curr[i * _I + j * _J + k];

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
                buf_rw[index] = w_curr[i * _I + j * _J + k].u_x;
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
                buf_rw[index] = w_curr[i * _I + j * _J + k].u_y;
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
                buf_rw[index] = w_curr[i * _I + j * _J + k].u_z;
                index++;
            }
        }
    }
    MPI_File_write_all(datafile_u_z, buf_rw, count_rw, MPI_DOUBLE, &status);
    this->current_iter_u_z += 1;
}

void Wave3d::_read_param_from_file()
{
    // Чтение параметров среды из файлов
#pragma region Recv_param
    // Чтение из файла параметра среды vp - скорость продольных волн
    MPI_File datafile_vp;
    std::string filename_vp = std::string("Vp.bin");
    char* file_vp = new char[filename_vp.length() + 1];
    strcpy_s(file_vp, filename_vp.length() + 1, filename_vp.c_str());

    MPI_File_open(communicator, file_vp, MPI_MODE_RDONLY, MPI_INFO_NULL, &datafile_vp);
    MPI_File_set_view(datafile_vp, 0, MPI_DOUBLE, subarray_rw, "native", MPI_INFO_NULL);
    MPI_File_read_all(datafile_vp, buf_rw, count_rw, MPI_DOUBLE, &status);

    long int index = 0;
    for (int i = starti_rw; i < endi_rw; i++)
    {
        for (int j = startj_rw; j < endj_rw; j++)
        {
            for (int k = 2 + PML_Size; k < I + 2 + PML_Size; k++)
            {
                v_p[i * _I + j * _J + k] = buf_rw[index];
                index++;
            }
        }
    }
    MPI_File_close(&datafile_vp);
    delete[] file_vp;

    // Чтение из файла параметра среды vs - скорость поперечных волн
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

    // Чтение из файла параметра среды rho - плотность
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
#pragma endregion

    // Параметры среды считаны из файла для исходной вычислительной области без PML.
    // Экстраполяция данных на PML - область и граничные точки.
#pragma region Extrapolate_param
    long int index_1;
    long int index_2;

    double _vp;
    double _vs;
    double rho;

    {
        ///////////////////////////////////////////////////////////////
        //// 
        //// +++++++++
        //// ++ ******
        //// ++ ******
        //// ++ ******
        ////
        ///////////////////////////////////////////////////////////////
        //if ((rank_coords[0] == 0) && (rank_coords[1] == 0))
        //{
        //    for (int i = 0; i < 2 + PML_Size; ++i)
        //    {
        //        for (int j = 0; j < 2 + PML_Size; ++j)
        //        {
        //            index_1 = i * _I + j * _J;
        //            index_2 = (2 + PML_Size) * _I + (2 + PML_Size) * _J + (2 + PML_Size);

        //            _vp = v_p[index_2];
        //            _vs = v_s[index_2];
        //            rho = Rho[index_2];

        //            for (int k = 0; k < 2 + PML_Size; ++k)
        //            {
        //                v_p[index_1] = _vp;
        //                v_s[index_1] = _vs;
        //                Rho[index_1] = rho;
        //                ++index_1;
        //            }

        //            for (int k = 2 + PML_Size; k < I + 2 + PML_Size; ++k)
        //            {
        //                v_p[index_1] = v_p[index_2];
        //                v_s[index_1] = v_s[index_2];
        //                Rho[index_1] = Rho[index_2];
        //                ++index_1;
        //                ++index_2;
        //            }

        //            --index_2;
        //            _vp = v_p[index_2];
        //            _vs = v_s[index_2];
        //            rho = Rho[index_2];

        //            for (int k = I + 2 + PML_Size; k < 4 + I + 2 * PML_Size; ++k)
        //            {
        //                v_p[index_1] = _vp;
        //                v_s[index_1] = _vs;
        //                Rho[index_1] = rho;
        //                ++index_1;
        //            }
        //        }

        //        for (int j = 2 + PML_Size; j < _size_j - 2; ++j)
        //        {
        //            index_1 = i * _I + j * _J;
        //            index_2 = (2 + PML_Size) * _I + j * _J + (2 + PML_Size);

        //            _vp = v_p[index_2];
        //            _vs = v_s[index_2];
        //            rho = Rho[index_2];

        //            for (int k = 0; k < 2 + PML_Size; ++k)
        //            {
        //                v_p[index_1] = _vp;
        //                v_s[index_1] = _vs;
        //                Rho[index_1] = rho;
        //                ++index_1;
        //            }

        //            for (int k = 2 + PML_Size; k < I + 2 + PML_Size; ++k)
        //            {
        //                v_p[index_1] = v_p[index_2];
        //                v_s[index_1] = v_s[index_2];
        //                Rho[index_1] = Rho[index_2];
        //                ++index_1;
        //                ++index_2;
        //            }

        //            --index_2;
        //            _vp = v_p[index_2];
        //            _vs = v_s[index_2];
        //            rho = Rho[index_2];

        //            for (int k = I + 2 + PML_Size; k < 4 + I + 2 * PML_Size; ++k)
        //            {
        //                v_p[index_1] = _vp;
        //                v_s[index_1] = _vs;
        //                Rho[index_1] = rho;
        //                ++index_1;
        //            }
        //        }
        //    }

        //    for (int i = 2 + PML_Size; i < _size_i - 2; ++i)
        //    {
        //        for (int j = 0; j < 2 + PML_Size; ++j)
        //        {
        //            index_1 = i * _I + j * _J;
        //            index_2 = i * _I + (2 + PML_Size) * _J + (2 + PML_Size);

        //            _vp = v_p[index_2];
        //            _vs = v_s[index_2];
        //            rho = Rho[index_2];

        //            for (int k = 0; k < 2 + PML_Size; ++k)
        //            {
        //                v_p[index_1] = _vp;
        //                v_s[index_1] = _vs;
        //                Rho[index_1] = rho;
        //                ++index_1;
        //            }

        //            for (int k = 2 + PML_Size; k < I + 2 + PML_Size; ++k)
        //            {
        //                v_p[index_1] = v_p[index_2];
        //                v_s[index_1] = v_s[index_2];
        //                Rho[index_1] = Rho[index_2];
        //                ++index_1;
        //                ++index_2;
        //            }

        //            --index_2;
        //            _vp = v_p[index_2];
        //            _vs = v_s[index_2];
        //            rho = Rho[index_2];

        //            for (int k = I + 2 + PML_Size; k < 4 + I + 2 * PML_Size; ++k)
        //            {
        //                v_p[index_1] = _vp;
        //                v_s[index_1] = _vs;
        //                Rho[index_1] = rho;
        //                ++index_1;
        //            }
        //        }

        //        for (int j = 2 + PML_Size; j < _size_j - 2; ++j)
        //        {
        //            index_1 = i * _I + j * _J;
        //            index_2 = i * _I + j * _J + 2 + PML_Size;

        //            _vp = v_p[index_2];
        //            _vs = v_s[index_2];
        //            rho = Rho[index_2];

        //            for (int k = 0; k < 2 + PML_Size; ++k)
        //            {
        //                v_p[index_1] = _vp;
        //                v_s[index_1] = _vs;
        //                Rho[index_1] = rho;
        //                ++index_1;
        //            }

        //            index_1 += I;
        //            index_2 += I - 1;
        //            _vp = v_p[index_2];
        //            _vs = v_s[index_2];
        //            rho = Rho[index_2];

        //            for (int k = 2 + I + PML_Size; k < 4 + I + 2 * PML_Size; ++k)
        //            {
        //                v_p[index_1] = _vp;
        //                v_s[index_1] = _vs;
        //                Rho[index_1] = rho;
        //                ++index_1;
        //            }
        //        }
        //    }
        //}
        //else
        ///////////////////////////////////////////////////////////////
        //// 
        //// ++ ******
        //// ++ ******
        //// ++ ******
        //// +++++++++
        ////
        ///////////////////////////////////////////////////////////////
        //if ((rank_coords[0] == dims[0] - 1) && (rank_coords[1] == 0))
        //{
        //    for (int i = 2; i < _size_i - PML_Size - 2; ++i)
        //    {
        //        for (int j = 0; j < 2 + PML_Size; ++j)
        //        {
        //            index_1 = i * _I + j * _J;
        //            index_2 = i * _I + (2 + PML_Size) * _J + 2 + PML_Size;

        //            _vp = v_p[index_2];
        //            _vs = v_s[index_2];
        //            rho = Rho[index_2];

        //            for (int k = 0; k < 2 + PML_Size; ++k)
        //            {
        //                v_p[index_1] = _vp;
        //                v_s[index_1] = _vs;
        //                Rho[index_1] = rho;
        //                ++index_1;
        //            }

        //            for (int k = 2 + PML_Size; k < I + 2 + PML_Size; ++k)
        //            {
        //                v_p[index_1] = v_p[index_2];
        //                v_s[index_1] = v_s[index_2];
        //                Rho[index_1] = Rho[index_2];
        //                ++index_1;
        //                ++index_2;
        //            }

        //            --index_2;
        //            _vp = v_p[index_2];
        //            _vs = v_s[index_2];
        //            rho = Rho[index_2];

        //            for (int k = I + 2 + PML_Size; k < 4 + I + 2 * PML_Size; ++k)
        //            {
        //                v_p[index_1] = _vp;
        //                v_s[index_1] = _vs;
        //                Rho[index_1] = rho;
        //                ++index_1;
        //            }
        //        }

        //        for (int j = 2 + PML_Size; j < _size_j - 2; ++j)
        //        {
        //            index_1 = i * _I + j * _J;
        //            index_2 = i * _I + j * _J + 2 + PML_Size;

        //            _vp = v_p[index_2];
        //            _vs = v_s[index_2];
        //            rho = Rho[index_2];

        //            for (int k = 0; k < 2 + PML_Size; ++k)
        //            {
        //                v_p[index_1] = _vp;
        //                v_s[index_1] = _vs;
        //                Rho[index_1] = rho;
        //                ++index_1;
        //            }

        //            index_1 += I;
        //            index_2 += I - 1;

        //            _vp = v_p[index_2];
        //            _vs = v_s[index_2];
        //            rho = Rho[index_2];

        //            for (int k = I + 2 + PML_Size; k < 4 + I + 2 * PML_Size; ++k)
        //            {
        //                v_p[index_1] = _vp;
        //                v_s[index_1] = _vs;
        //                Rho[index_1] = rho;
        //                ++index_1;
        //            }
        //        }
        //    }

        //    for (int i = _size_i - PML_Size - 2; i < _size_i; ++i)
        //    {
        //        for (int j = 0; j < 2 + PML_Size; ++j)
        //        {
        //            index_1 = i * _I + j * _J;
        //            index_2 = (_size_i - 3 - PML_Size) * _I + (2 + PML_Size) * _J + 2 + PML_Size;

        //            _vp = v_p[index_2];
        //            _vs = v_s[index_2];
        //            rho = Rho[index_2];

        //            for (int k = 0; k < 2 + PML_Size; ++k)
        //            {
        //                v_p[index_1] = _vp;
        //                v_s[index_1] = _vs;
        //                Rho[index_1] = rho;
        //                ++index_1;
        //            }

        //            for (int k = 2 + PML_Size; k < I + 2 + PML_Size; ++k)
        //            {
        //                v_p[index_1] = v_p[index_2];
        //                v_s[index_1] = v_s[index_2];
        //                Rho[index_1] = Rho[index_2];
        //                ++index_1;
        //                ++index_2;
        //            }

        //            --index_2;

        //            _vp = v_p[index_2];
        //            _vs = v_s[index_2];
        //            rho = Rho[index_2];

        //            for (int k = I + 2 + PML_Size; k < 4 + I + 2 * PML_Size; ++k)
        //            {
        //                v_p[index_1] = _vp;
        //                v_s[index_1] = _vs;
        //                Rho[index_1] = rho;
        //                ++index_1;
        //            }
        //        }

        //        for (int j = 2 + PML_Size; j < _size_j - 2; ++j)
        //        {
        //            index_1 = i * _I + j * _J;
        //            index_2 = (_size_i - 3 - PML_Size) * _I + j * _J + 2 + PML_Size;

        //            _vp = v_p[index_2];
        //            _vs = v_s[index_2];
        //            rho = Rho[index_2];

        //            for (int k = 0; k < 2 + PML_Size; ++k)
        //            {
        //                v_p[index_1] = _vp;
        //                v_s[index_1] = _vs;
        //                Rho[index_1] = rho;
        //                ++index_1;
        //            }

        //            for (int k = 2 + PML_Size; k < I + 2 + PML_Size; ++k)
        //            {
        //                v_p[index_1] = v_p[index_2];
        //                v_s[index_1] = v_s[index_2];
        //                Rho[index_1] = Rho[index_2];
        //                ++index_1;
        //                ++index_2;
        //            }

        //            --index_2;
        //            _vp = v_p[index_2];
        //            _vs = v_s[index_2];
        //            rho = Rho[index_2];

        //            for (int k = I + 2 + PML_Size; k < 4 + I + 2 * PML_Size; ++k)
        //            {
        //                v_p[index_1] = _vp;
        //                v_s[index_1] = _vs;
        //                Rho[index_1] = rho;
        //                ++index_1;
        //            }
        //        }
        //    }
        //}
        //else
        ///////////////////////////////////////////////////////////////
        //// 
        //// +++++++++
        //// ****** ++
        //// ****** ++
        //// ****** ++
        ////
        ///////////////////////////////////////////////////////////////
        //if ((rank_coords[0] == 0) && (rank_coords[1] == dims[1] - 1))
        //{
        //    for (int i = 0; i < 2 + PML_Size; ++i)
        //    {
        //        for (int j = 2; j < _size_j - PML_Size - 2; ++j)
        //        {
        //            index_1 = i * _I + j * _J;
        //            index_2 = (2 + PML_Size) * _I + j * _J + 2 + PML_Size;

        //            _vp = v_p[index_2];
        //            _vs = v_s[index_2];
        //            rho = Rho[index_2];

        //            for (int k = 0; k < 2 + PML_Size; ++k)
        //            {
        //                v_p[index_1] = _vp;
        //                v_s[index_1] = _vs;
        //                Rho[index_1] = rho;
        //                ++index_1;
        //            }

        //            for (int k = 2 + PML_Size; k < I + 2 + PML_Size; ++k)
        //            {
        //                v_p[index_1] = v_p[index_2];
        //                v_s[index_1] = v_s[index_2];
        //                Rho[index_1] = Rho[index_2];
        //                ++index_1;
        //                ++index_2;
        //            }

        //            --index_2;
        //            _vp = v_p[index_2];
        //            _vs = v_s[index_2];
        //            rho = Rho[index_2];

        //            for (int k = I + 2 + PML_Size; k < 4 + I + 2 * PML_Size; ++k)
        //            {
        //                v_p[index_1] = _vp;
        //                v_s[index_1] = _vs;
        //                Rho[index_1] = rho;
        //                ++index_1;
        //            }
        //        }

        //        for (int j = _size_j - PML_Size - 2; j < _size_j; ++j)
        //        {
        //            index_1 = i * _I + j * _J;
        //            index_2 = (2 + PML_Size) * _I + (_size_j - 3 - PML_Size) * _J + 2 + PML_Size;

        //            _vp = v_p[index_2];
        //            _vs = v_s[index_2];
        //            rho = Rho[index_2];

        //            for (int k = 0; k < 2 + PML_Size; ++k)
        //            {
        //                v_p[index_1] = _vp;
        //                v_s[index_1] = _vs;
        //                Rho[index_1] = rho;
        //                ++index_1;
        //            }

        //            for (int k = 2 + PML_Size; k < I + 2 + PML_Size; ++k)
        //            {
        //                v_p[index_1] = v_p[index_2];
        //                v_s[index_1] = v_s[index_2];
        //                Rho[index_1] = Rho[index_2];
        //                ++index_1;
        //                ++index_2;
        //            }

        //            --index_2;
        //            _vp = v_p[index_2];
        //            _vs = v_s[index_2];
        //            rho = Rho[index_2];

        //            for (int k = I + 2 + PML_Size; k < 4 + I + 2 * PML_Size; ++k)
        //            {
        //                v_p[index_1] = _vp;
        //                v_s[index_1] = _vs;
        //                Rho[index_1] = rho;
        //                ++index_1;
        //            }
        //        }
        //    }

        //    for (int i = 2 + PML_Size; i < _size_i - 2; ++i)
        //    {
        //        for (int j = 2; j < _size_j - PML_Size - 2; ++j)
        //        {
        //            index_1 = i * _I + j * _J;
        //            index_2 = i * _I + j * _J + 2 + PML_Size;

        //            _vp = v_p[index_2];
        //            _vs = v_s[index_2];
        //            rho = Rho[index_2];

        //            for (int k = 0; k < 2 + PML_Size; ++k)
        //            {
        //                v_p[index_1] = _vp;
        //                v_s[index_1] = _vs;
        //                Rho[index_1] = rho;
        //                ++index_1;
        //            }

        //            index_1 += I;
        //            index_2 += I - 1;
        //            _vp = v_p[index_2];
        //            _vs = v_s[index_2];
        //            rho = Rho[index_2];

        //            for (int k = I + 2 + PML_Size; k < 4 + I + 2 * PML_Size; ++k)
        //            {
        //                v_p[index_1] = _vp;
        //                v_s[index_1] = _vs;
        //                Rho[index_1] = rho;
        //                ++index_1;
        //            }
        //        }

        //        for (int j = _size_j - PML_Size - 2; j < _size_j; ++j)
        //        {
        //            index_1 = i * _I + j * _J;
        //            index_2 = i * _I + (_size_j - 3 - PML_Size) * _J + 2 + PML_Size;

        //            _vp = v_p[index_2];
        //            _vs = v_s[index_2];
        //            rho = Rho[index_2];

        //            for (int k = 0; k < 2 + PML_Size; ++k)
        //            {
        //                v_p[index_1] = _vp;
        //                v_s[index_1] = _vs;
        //                Rho[index_1] = rho;
        //                ++index_1;
        //            }

        //            for (int k = 2 + PML_Size; k < I + 2 + PML_Size; ++k)
        //            {
        //                v_p[index_1] = v_p[index_2];
        //                v_s[index_1] = v_s[index_2];
        //                Rho[index_1] = Rho[index_2];
        //                ++index_1;
        //                ++index_2;
        //            }

        //            --index_2;
        //            _vp = v_p[index_2];
        //            _vs = v_s[index_2];
        //            rho = Rho[index_2];

        //            for (int k = I + 2 + PML_Size; k < 4 + I + 2 * PML_Size; ++k)
        //            {
        //                v_p[index_1] = _vp;
        //                v_s[index_1] = _vs;
        //                Rho[index_1] = rho;
        //                ++index_1;
        //            }
        //        }
        //    }
        //}
        //else
        ///////////////////////////////////////////////////////////////
        //// 
        //// ****** ++
        //// ****** ++
        //// ****** ++
        //// +++++++++
        ////
        ///////////////////////////////////////////////////////////////
        //if ((rank_coords[0] == dims[0] - 1) && (rank_coords[1] == dims[1] - 1))
        //{
        //    for (int i = 2; i < _size_i - PML_Size - 2; ++i)
        //    {
        //        for (int j = 2; j < _size_j - PML_Size - 2; ++j)
        //        {
        //            index_1 = i * _I + j * _J;
        //            index_2 = i * _I + j * _J + 2 + PML_Size;

        //            _vp = v_p[index_2];
        //            _vs = v_s[index_2];
        //            rho = Rho[index_2];

        //            for (int k = 0; k < 2 + PML_Size; ++k)
        //            {
        //                v_p[index_1] = _vp;
        //                v_s[index_1] = _vs;
        //                Rho[index_1] = rho;
        //                ++index_1;
        //            }

        //            index_1 += I;
        //            index_2 += I - 1;
        //            _vp = v_p[index_2];
        //            _vs = v_s[index_2];
        //            rho = Rho[index_2];

        //            for (int k = I + 2 + PML_Size; k < 4 + I + 2 * PML_Size; ++k)
        //            {
        //                v_p[index_1] = _vp;
        //                v_s[index_1] = _vs;
        //                Rho[index_1] = rho;
        //                ++index_1;
        //            }
        //        }

        //        for (int j = _size_j - PML_Size - 2; j < _size_j; ++j)
        //        {
        //            index_1 = i * _I + j * _J;
        //            index_2 = i * _I + (_size_j - 3 - PML_Size) * _J + 2 + PML_Size;

        //            _vp = v_p[index_2];
        //            _vs = v_s[index_2];
        //            rho = Rho[index_2];

        //            for (int k = 0; k < 2 + PML_Size; ++k)
        //            {
        //                v_p[index_1] = _vp;
        //                v_s[index_1] = _vs;
        //                Rho[index_1] = rho;
        //                ++index_1;
        //            }

        //            for (int k = 2 + PML_Size; k < I + 2 + PML_Size; ++k)
        //            {
        //                v_p[index_1] = v_p[index_2];
        //                v_s[index_1] = v_s[index_2];
        //                Rho[index_1] = Rho[index_2];
        //                ++index_1;
        //                ++index_2;
        //            }

        //            --index_2;
        //            _vp = v_p[index_2];
        //            _vs = v_s[index_2];
        //            rho = Rho[index_2];

        //            for (int k = I + 2 + PML_Size; k < 4 + I + 2 * PML_Size; ++k)
        //            {
        //                v_p[index_1] = _vp;
        //                v_s[index_1] = _vs;
        //                Rho[index_1] = rho;
        //                ++index_1;
        //            }
        //        }
        //    }

        //    for (int i = _size_i - PML_Size - 2; i < _size_i; ++i)
        //    {
        //        for (int j = 2; j < _size_j - PML_Size - 2; ++j)
        //        {
        //            index_1 = i * _I + j * _J;
        //            index_2 = (_size_i - 3 - PML_Size) * _I + j * _J + 2 + PML_Size;

        //            _vp = v_p[index_2];
        //            _vs = v_s[index_2];
        //            rho = Rho[index_2];

        //            for (int k = 0; k < 2 + PML_Size; ++k)
        //            {
        //                v_p[index_1] = _vp;
        //                v_s[index_1] = _vs;
        //                Rho[index_1] = rho;
        //                ++index_1;
        //            }

        //            for (int k = 2 + PML_Size; k < I + 2 + PML_Size; ++k)
        //            {
        //                v_p[index_1] = v_p[index_2];
        //                v_s[index_1] = v_s[index_2];
        //                Rho[index_1] = Rho[index_2];
        //                ++index_1;
        //                ++index_2;
        //            }

        //            --index_2;
        //            _vp = v_p[index_2];
        //            _vs = v_s[index_2];
        //            rho = Rho[index_2];

        //            for (int k = I + 2 + PML_Size; k < 4 + I + 2 * PML_Size; ++k)
        //            {
        //                v_p[index_1] = _vp;
        //                v_s[index_1] = _vs;
        //                Rho[index_1] = rho;
        //                ++index_1;
        //            }
        //        }

        //        for (int j = _size_j - PML_Size - 2; j < _size_j; ++j)
        //        {
        //            index_1 = i * _I + j * _J;
        //            index_2 = (_size_i - 3 - PML_Size) * _I + (_size_j - 3 - PML_Size) * _J + 2 + PML_Size;

        //            _vp = v_p[index_2];
        //            _vs = v_s[index_2];
        //            rho = Rho[index_2];

        //            for (int k = 0; k < 2 + PML_Size; ++k)
        //            {
        //                v_p[index_1] = _vp;
        //                v_s[index_1] = _vs;
        //                Rho[index_1] = rho;
        //                ++index_1;
        //            }

        //            for (int k = 2 + PML_Size; k < I + 2 + PML_Size; ++k)
        //            {
        //                v_p[index_1] = v_p[index_2];
        //                v_s[index_1] = v_s[index_2];
        //                Rho[index_1] = Rho[index_2];
        //                ++index_1;
        //                ++index_2;
        //            }

        //            --index_2;
        //            _vp = v_p[index_2];
        //            _vs = v_s[index_2];
        //            rho = Rho[index_2];

        //            for (int k = I + 2 + PML_Size; k < 4 + I + 2 * PML_Size; ++k)
        //            {
        //                v_p[index_1] = _vp;
        //                v_s[index_1] = _vs;
        //                Rho[index_1] = rho;
        //                ++index_1;
        //            }
        //        }
        //    }
        //}
        //else
        ///////////////////////////////////////////////////////////////
        //// 
        //// ++ ******
        //// ++ ******
        //// ++ ******
        //// ++ ******
        ////
        ///////////////////////////////////////////////////////////////
        //if ((rank_coords[0] != 0) && (rank_coords[0] != dims[0] - 1) && (rank_coords[1] == 0))
        //{
        //    for (int i = 2; i < _size_i - 2; ++i)
        //    {
        //        for (int j = 0; j < 2 + PML_Size; ++j)
        //        {
        //            index_1 = i * _I + j * _J;
        //            index_2 = i * _I + (2 + PML_Size) * _J + 2 + PML_Size;

        //            _vp = v_p[index_2];
        //            _vs = v_s[index_2];
        //            rho = Rho[index_2];

        //            for (int k = 0; k < 2 + PML_Size; ++k)
        //            {
        //                v_p[index_1] = _vp;
        //                v_s[index_1] = _vs;
        //                Rho[index_1] = rho;
        //                ++index_1;
        //            }

        //            for (int k = 2 + PML_Size; k < I + 2 + PML_Size; ++k)
        //            {
        //                v_p[index_1] = v_p[index_2];
        //                v_s[index_1] = v_s[index_2];
        //                Rho[index_1] = Rho[index_2];
        //                ++index_1;
        //                ++index_2;
        //            }

        //            --index_2;
        //            _vp = v_p[index_2];
        //            _vs = v_s[index_2];
        //            rho = Rho[index_2];

        //            for (int k = I + 2 + PML_Size; k < 4 + I + 2 * PML_Size; ++k)
        //            {
        //                v_p[index_1] = _vp;
        //                v_s[index_1] = _vs;
        //                Rho[index_1] = rho;
        //                ++index_1;
        //            }
        //        }

        //        for (int j = 2 + PML_Size; j < _size_j - 2; ++j)
        //        {
        //            index_1 = i * _I + j * _J;
        //            index_2 = i * _I + j * _J + 2 + PML_Size;

        //            _vp = v_p[index_2];
        //            _vs = v_s[index_2];
        //            rho = Rho[index_2];

        //            for (int k = 0; k < 2 + PML_Size; ++k)
        //            {
        //                v_p[index_1] = _vp;
        //                v_s[index_1] = _vs;
        //                Rho[index_1] = rho;
        //                ++index_1;
        //            }

        //            index_1 += I;
        //            index_2 += I - 1;
        //            _vp = v_p[index_2];
        //            _vs = v_s[index_2];
        //            rho = Rho[index_2];

        //            for (int k = I + 2 + PML_Size; k < 4 + I + 2 * PML_Size; ++k)
        //            {
        //                v_p[index_1] = _vp;
        //                v_s[index_1] = _vs;
        //                Rho[index_1] = rho;
        //                ++index_1;
        //            }
        //        }
        //    }
        //}
        //else
        ///////////////////////////////////////////////////////////////
        //// 
        //// ****** ++
        //// ****** ++ 
        //// ****** ++
        //// ****** ++
        ////
        ///////////////////////////////////////////////////////////////
        //if ((rank_coords[0] != 0) && (rank_coords[0] != dims[0] - 1) && (rank_coords[1] == dims[1] - 1))
        //{
        //    for (int i = 2; i < _size_i - 2; ++i)
        //    {
        //        for (int j = 2; j < _size_j - PML_Size - 2; ++j)
        //        {
        //            index_1 = i * _I + j * _J;
        //            index_2 = i * _I + j * _J + 2 + PML_Size;

        //            _vp = v_p[index_2];
        //            _vs = v_s[index_2];
        //            rho = Rho[index_2];

        //            for (int k = 0; k < 2 + PML_Size; ++k)
        //            {
        //                v_p[index_1] = _vp;
        //                v_s[index_1] = _vs;
        //                Rho[index_1] = rho;
        //                ++index_1;
        //            }

        //            index_1 += I;
        //            index_2 += I - 1;
        //            _vp = v_p[index_2];
        //            _vs = v_s[index_2];
        //            rho = Rho[index_2];

        //            for (int k = I + 2 + PML_Size; k < 4 + I + 2 * PML_Size; ++k)
        //            {
        //                v_p[index_1] = _vp;
        //                v_s[index_1] = _vs;
        //                Rho[index_1] = rho;
        //                ++index_1;
        //            }
        //        }

        //        for (int j = _size_j - PML_Size - 2; j < _size_j; ++j)
        //        {
        //            index_1 = i * _I + j * _J;
        //            index_2 = i * _I + (_size_j - 3 - PML_Size) * _J + 2 + PML_Size;

        //            _vp = v_p[index_2];
        //            _vs = v_s[index_2];
        //            rho = Rho[index_2];

        //            for (int k = 0; k < 2 + PML_Size; ++k)
        //            {
        //                v_p[index_1] = _vp;
        //                v_s[index_1] = _vs;
        //                Rho[index_1] = rho;
        //                ++index_1;
        //            }

        //            for (int k = 2 + PML_Size; k < I + 2 + PML_Size; ++k)
        //            {
        //                v_p[index_1] = v_p[index_2];
        //                v_s[index_1] = v_s[index_2];
        //                Rho[index_1] = Rho[index_2];
        //                ++index_1;
        //                ++index_2;
        //            }

        //            --index_2;
        //            _vp = v_p[index_2];
        //            _vs = v_s[index_2];
        //            rho = Rho[index_2];

        //            for (int k = I + 2 + PML_Size; k < 4 + I + 2 * PML_Size; ++k)
        //            {
        //                v_p[index_1] = _vp;
        //                v_s[index_1] = _vs;
        //                Rho[index_1] = rho;
        //                ++index_1;
        //            }
        //        }
        //    }
        //}
        //else
        ///////////////////////////////////////////////////////////////
        //// 
        //// ++++++++
        //// ******** 
        //// ********
        //// ********
        ////
        ///////////////////////////////////////////////////////////////
        //if ((rank_coords[1] != 0) && (rank_coords[1] != dims[1] - 1) && (rank_coords[0] == 0))
        //{
        //    for (int j = 2; j < _size_j - 2; ++j)
        //    {
        //        for (int i = 0; i < 2 + PML_Size; ++i)
        //        {
        //            index_1 = i * _I + j * _J;
        //            index_2 = (2 + PML_Size) * _I + j * _J + 2 + PML_Size;

        //            _vp = v_p[index_2];
        //            _vs = v_s[index_2];
        //            rho = Rho[index_2];

        //            for (int k = 0; k < 2 + PML_Size; ++k)
        //            {
        //                v_p[index_1] = _vp;
        //                v_s[index_1] = _vs;
        //                Rho[index_1] = rho;
        //                ++index_1;
        //            }

        //            for (int k = 2 + PML_Size; k < I + 2 + PML_Size; ++k)
        //            {
        //                v_p[index_1] = v_p[index_2];
        //                v_s[index_1] = v_s[index_2];
        //                Rho[index_1] = Rho[index_2];
        //                ++index_1;
        //                ++index_2;
        //            }

        //            --index_2;
        //            _vp = v_p[index_2];
        //            _vs = v_s[index_2];
        //            rho = Rho[index_2];

        //            for (int k = I + 2 + PML_Size; k < 4 + I + 2 * PML_Size; ++k)
        //            {
        //                v_p[index_1] = _vp;
        //                v_s[index_1] = _vs;
        //                Rho[index_1] = rho;
        //                ++index_1;
        //            }
        //        }

        //        for (int i = 2 + PML_Size; i < _size_i - 2; ++i)
        //        {
        //            index_1 = i * _I + j * _J;
        //            index_2 = i * _I + j * _J + 2 + PML_Size;

        //            _vp = v_p[index_2];
        //            _vs = v_s[index_2];
        //            rho = Rho[index_2];

        //            for (int k = 0; k < 2 + PML_Size; ++k)
        //            {
        //                v_p[index_1] = _vp;
        //                v_s[index_1] = _vs;
        //                Rho[index_1] = rho;
        //                ++index_1;
        //            }

        //            index_1 += I;
        //            index_2 += I - 1;
        //            _vp = v_p[index_2];
        //            _vs = v_s[index_2];
        //            rho = Rho[index_2];

        //            for (int k = I + 2 + PML_Size; k < 4 + I + 2 * PML_Size; ++k)
        //            {
        //                v_p[index_1] = _vp;
        //                v_s[index_1] = _vs;
        //                Rho[index_1] = rho;
        //                ++index_1;
        //            }
        //        }
        //    }
        //}
        //else
        ///////////////////////////////////////////////////////////////
        //// 
        //// ******** 
        //// ********
        //// ********
        //// ++++++++
        ////
        ///////////////////////////////////////////////////////////////
        //if ((rank_coords[1] != 0) && (rank_coords[1] != dims[1] - 1) && (rank_coords[0] == dims[0] - 1))
        //{
        //    for (int j = 2; j < _size_j - 2; ++j)
        //    {
        //        for (int i = 2; i < _size_i - PML_Size - 2; ++i)
        //        {
        //            index_1 = i * _I + j * _J;
        //            index_2 = i * _I + j * _J + 2 + PML_Size;

        //            _vp = v_p[index_2];
        //            _vs = v_s[index_2];
        //            rho = Rho[index_2];

        //            for (int k = 0; k < 2 + PML_Size; ++k)
        //            {
        //                v_p[index_1] = _vp;
        //                v_s[index_1] = _vs;
        //                Rho[index_1] = rho;
        //                ++index_1;
        //            }

        //            index_1 += I;
        //            index_2 += I - 1;
        //            _vp = v_p[index_2];
        //            _vs = v_s[index_2];
        //            rho = Rho[index_2];

        //            for (int k = I + 2 + PML_Size; k < 4 + I + 2 * PML_Size; ++k)
        //            {
        //                v_p[index_1] = _vp;
        //                v_s[index_1] = _vs;
        //                Rho[index_1] = rho;
        //                ++index_1;
        //            }
        //        }

        //        for (int i = _size_i - PML_Size - 2; i < _size_i; ++i)
        //        {
        //            index_1 = i * _I + j * _J;
        //            index_2 = (_size_i - 3 - PML_Size) * _I + j * _J + 2 + PML_Size;

        //            _vp = v_p[index_2];
        //            _vs = v_s[index_2];
        //            rho = Rho[index_2];

        //            for (int k = 0; k < 2 + PML_Size; ++k)
        //            {
        //                v_p[index_1] = _vp;
        //                v_s[index_1] = _vs;
        //                Rho[index_1] = rho;
        //                ++index_1;
        //            }

        //            for (int k = 2 + PML_Size; k < I + 2 + PML_Size; ++k)
        //            {
        //                v_p[index_1] = v_p[index_2];
        //                v_s[index_1] = v_s[index_2];
        //                Rho[index_1] = Rho[index_2];
        //                ++index_1;
        //                ++index_2;
        //            }

        //            --index_2;
        //            _vp = v_p[index_2];
        //            _vs = v_s[index_2];
        //            rho = Rho[index_2];

        //            for (int k = I + 2 + PML_Size; k < 4 + I + 2 * PML_Size; ++k)
        //            {
        //                v_p[index_1] = _vp;
        //                v_s[index_1] = _vs;
        //                Rho[index_1] = rho;
        //                ++index_1;
        //            }
        //        }
        //    }
        //}
        //else
        //for (long int i = 2; i < _size_i - 2; ++i)
        //{
        //    for (long int j = 2; j < _size_j - 2; ++j)
        //    {
        //        for (long int k = 0; k < 2 + PML_Size; k++)
        //        {
        //            v_p[i * _I + j * _J + k] = v_p[i * _I + j * _J + 2 + PML_Size];
        //            v_s[i * _I + j * _J + k] = v_s[i * _I + j * _J + 2 + PML_Size];
        //            Rho[i * _I + j * _J + k] = Rho[i * _I + j * _J + 2 + PML_Size];
        //        }

        //        for (long int k = I + 2 + PML_Size; k < 4 + I + 2 * PML_Size; k++)
        //        {
        //            v_p[i * _I + j * _J + k] = v_p[i * _I + j * _J + _size_k - 3 - PML_Size];
        //            v_s[i * _I + j * _J + k] = v_s[i * _I + j * _J + _size_k - 3 - PML_Size];
        //            Rho[i * _I + j * _J + k] = Rho[i * _I + j * _J + _size_k - 3 - PML_Size];
        //        }
        //    }
        //}
    }

    for (long int i = 0; i < starti_rw; ++i)
    {
        for (long int j = 0; j < startj_rw; ++j)
        {
            index_1 = i * _I + j * _J;
            index_2 = starti_rw * _I + startj_rw * _J + 2 + PML_Size;

            _vp = v_p[index_2];
            _vs = v_s[index_2];
            rho = Rho[index_2];

            for (long int k = 0; k < 2 + PML_Size; ++k)
            {
                v_p[index_1] = _vp;
                v_s[index_1] = _vs;
                Rho[index_1] = rho;
                ++index_1;
            }

            for (long int k = 2 + PML_Size; k < I + 2 + PML_Size; ++k)
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

        for (long int j = startj_rw; j < endj_rw; ++j)
        {
            index_1 = i * _I + j * _J;
            index_2 = starti_rw * _I + j * _J + 2 + PML_Size;

            _vp = v_p[index_2];
            _vs = v_s[index_2];
            rho = Rho[index_2];

            for (long int k = 0; k < 2 + PML_Size; ++k)
            {
                v_p[index_1] = _vp;
                v_s[index_1] = _vs;
                Rho[index_1] = rho;
                ++index_1;
            }

            for (long int k = 2 + PML_Size; k < I + 2 + PML_Size; ++k)
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

        for (long int j = endj_rw; j < _size_j; ++j)
        {
            index_1 = i * _I + j * _J;
            index_2 = starti_rw * _I + (endj_rw - 1) * _J + 2 + PML_Size;

            _vp = v_p[index_2];
            _vs = v_s[index_2];
            rho = Rho[index_2];

            for (long int k = 0; k < 2 + PML_Size; ++k)
            {
                v_p[index_1] = _vp;
                v_s[index_1] = _vs;
                Rho[index_1] = rho;
                ++index_1;
            }

            for (long int k = 2 + PML_Size; k < I + 2 + PML_Size; ++k)
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

    for (long int i = starti_rw; i < endi_rw; ++i)
    {
        for (long int j = 0; j < startj_rw; ++j)
        {
            index_1 = i * _I + j * _J;
            index_2 = i * _I + startj_rw * _J + 2 + PML_Size;

            _vp = v_p[index_2];
            _vs = v_s[index_2];
            rho = Rho[index_2];

            for (long int k = 0; k < 2 + PML_Size; ++k)
            {
                v_p[index_1] = _vp;
                v_s[index_1] = _vs;
                Rho[index_1] = rho;
                ++index_1;
            }

            for (long int k = 2 + PML_Size; k < I + 2 + PML_Size; ++k)
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

        for (long int j = startj_rw; j < endj_rw; ++j)
        {
            index_1 = i * _I + j * _J;
            index_2 = i * _I + j * _J + 2 + PML_Size;

            _vp = v_p[index_2];
            _vs = v_s[index_2];
            rho = Rho[index_2];

            for (long int k = 0; k < 2 + PML_Size; ++k)
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

        for (long int j = endj_rw; j < _size_j; ++j)
        {
            index_1 = i * _I + j * _J;
            index_2 = i * _I + (endj_rw - 1) * _J + 2 + PML_Size;

            _vp = v_p[index_2];
            _vs = v_s[index_2];
            rho = Rho[index_2];

            for (long int k = 0; k < 2 + PML_Size; ++k)
            {
                v_p[index_1] = _vp;
                v_s[index_1] = _vs;
                Rho[index_1] = rho;
                ++index_1;
            }

            for (long int k = 2 + PML_Size; k < I + 2 + PML_Size; ++k)
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

    for (long int i = endi_rw; i < _size_i; ++i)
    {
        for (long int j = 0; j < startj_rw; ++j)
        {
            index_1 = i * _I + j * _J;
            index_2 = (endi_rw - 1) * _I + startj_rw * _J + 2 + PML_Size;

            _vp = v_p[index_2];
            _vs = v_s[index_2];
            rho = Rho[index_2];

            for (long int k = 0; k < 2 + PML_Size; ++k)
            {
                v_p[index_1] = _vp;
                v_s[index_1] = _vs;
                Rho[index_1] = rho;
                ++index_1;
            }

            for (long int k = 2 + PML_Size; k < I + 2 + PML_Size; ++k)
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

        for (long int j = startj_rw; j < endj_rw; ++j)
        {
            index_1 = i * _I + j * _J;
            index_2 = (endi_rw - 1) * _I + j * _J + 2 + PML_Size;

            _vp = v_p[index_2];
            _vs = v_s[index_2];
            rho = Rho[index_2];

            for (long int k = 0; k < 2 + PML_Size; ++k)
            {
                v_p[index_1] = _vp;
                v_s[index_1] = _vs;
                Rho[index_1] = rho;
                ++index_1;
            }

            for (long int k = 2 + PML_Size; k < I + 2 + PML_Size; ++k)
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

        for (long int j = endj_rw; j < _size_j; ++j)
        {
            index_1 = i * _I + j * _J;
            index_2 = (endi_rw - 1) * _I + (endj_rw - 1) * _J + 2 + PML_Size;

            _vp = v_p[index_2];
            _vs = v_s[index_2];
            rho = Rho[index_2];

            for (long int k = 0; k < 2 + PML_Size; ++k)
            {
                v_p[index_1] = _vp;
                v_s[index_1] = _vs;
                Rho[index_1] = rho;
                ++index_1;
            }

            for (long int k = 2 + PML_Size; k < I + 2 + PML_Size; ++k)
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
#pragma endregion

 #pragma region MPI_Type_create_subarray_param
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

    // Левая граница для отправки
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

    // Левая граница для чтения
    starts[1] = 0;
    MPI_Type_create_subarray(3, gsizes, lsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &left_subaray_recv_param);
    MPI_Type_commit(&left_subaray_recv_param);

    //printf("left_subaray_recv_param\n    gsizes:\n        %d\n        %d\n        %d\n", gsizes[0], gsizes[1], gsizes[2]);
    //printf("    lsizes:\n        %d\n        %d\n        %d\n", lsizes[0], lsizes[1], lsizes[2]);
    //printf("    starts:\n        %d\n        %d\n        %d\n", starts[0], starts[1], starts[2]);

    // правая граница для отправки
    starts[1] = _size_j - 4;
    MPI_Type_create_subarray(3, gsizes, lsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &right_subaray_send_param);
    MPI_Type_commit(&right_subaray_send_param);

    //printf("right_subaray_send_param\n    gsizes:\n        %d\n        %d\n        %d\n", gsizes[0], gsizes[1], gsizes[2]);
    //printf("    lsizes:\n        %d\n        %d\n        %d\n", lsizes[0], lsizes[1], lsizes[2]);
    //printf("    starts:\n        %d\n        %d\n        %d\n", starts[0], starts[1], starts[2]);

    // правая граница для чтения
    starts[1] = _size_j - 2;
    MPI_Type_create_subarray(3, gsizes, lsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &right_subaray_recv_param);
    MPI_Type_commit(&right_subaray_recv_param);

    //printf("right_subaray_recv_param\n    gsizes:\n        %d\n        %d\n        %d\n", gsizes[0], gsizes[1], gsizes[2]);
    //printf("    lsizes:\n        %d\n        %d\n        %d\n", lsizes[0], lsizes[1], lsizes[2]);
    //printf("    starts:\n        %d\n        %d\n        %d\n", starts[0], starts[1], starts[2]);


    // верхняя граница для отправки
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

    // верхняя граница для чтения
    starts[0] = 0;
    MPI_Type_create_subarray(3, gsizes, lsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &upper_subaray_recv_param);
    MPI_Type_commit(&upper_subaray_recv_param);

    //printf("upper_subaray_recv_param\n    gsizes:\n        %d\n        %d\n        %d\n", gsizes[0], gsizes[1], gsizes[2]);
    //printf("    lsizes:\n        %d\n        %d\n        %d\n", lsizes[0], lsizes[1], lsizes[2]);
    //printf("    starts:\n        %d\n        %d\n        %d\n", starts[0], starts[1], starts[2]);

    // нижняя граница для отправки
    starts[0] = _size_i - 4;
    MPI_Type_create_subarray(3, gsizes, lsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &lower_subaray_send_param);
    MPI_Type_commit(&lower_subaray_send_param);

    //printf("lower_subaray_send_param\n    gsizes:\n        %d\n        %d\n        %d\n", gsizes[0], gsizes[1], gsizes[2]);
    //printf("    lsizes:\n        %d\n        %d\n        %d\n", lsizes[0], lsizes[1], lsizes[2]);
    //printf("    starts:\n        %d\n        %d\n        %d\n", starts[0], starts[1], starts[2]);

    // нижняя граница для чтения
    starts[0] = _size_i - 2;
    MPI_Type_create_subarray(3, gsizes, lsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &lower_subaray_recv_param);
    MPI_Type_commit(&lower_subaray_recv_param);

    //printf("lower_subaray_recv_param\n    gsizes:\n        %d\n        %d\n        %d\n", gsizes[0], gsizes[1], gsizes[2]);
    //printf("    lsizes:\n        %d\n        %d\n        %d\n", lsizes[0], lsizes[1], lsizes[2]);
    //printf("    starts:\n        %d\n        %d\n        %d\n", starts[0], starts[1], starts[2]);
#pragma endregion

    // Объмен данных между потоками: пересылка параметров среды по оси x.
#pragma region MPI_Sendrecv_param_X
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
#pragma endregion

    // Объмен данных между потоками: пересылка параметров среды по оси y.
#pragma region MPI_Sendrecv_param_Y
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
#pragma endregion

#pragma region MPI_Type_free_subarray_param
    MPI_Type_free(&left_subaray_send_param);
    MPI_Type_free(&left_subaray_recv_param);

    MPI_Type_free(&right_subaray_send_param);
    MPI_Type_free(&right_subaray_recv_param);

    MPI_Type_free(&upper_subaray_send_param);
    MPI_Type_free(&upper_subaray_recv_param);

    MPI_Type_free(&lower_subaray_send_param);
    MPI_Type_free(&lower_subaray_recv_param);
#pragma endregion
}

Wave3d::Wave3d(int I, double T, std::function<double(double, double, double, double)> f)
{
    // Инициализация копмпонент для параллельной работы программы
#pragma region MPI init
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Декомпозиция области. Переход к двумерной декартовой топологии процессов
    ierror = MPI_Dims_create(size, 2, dims);
    if (ierror != MPI_SUCCESS)
        printf("MPI_Dims_create error!");

    // Декомпозиция области. Создание нового комуникатора
    ierror = MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &communicator);
    if (ierror != MPI_SUCCESS)
        printf("MPI_Cart_create error!");

    // Определение координат в двумерной топологии
    ierror = MPI_Comm_rank(communicator, &rank);
    ierror = MPI_Cart_coords(communicator, rank, 2, rank_coords);

    ///////////////////////////////////////////////////////
    // Расположение процессов в плоскости XY
    //    +---+---+---+
    //    | 0 | 1 | 2 |                   upper        |         upper = 1           |         upper = -1
    //    +---+---+---+                   +---+        |           +---+             |           +---+
    //    | 3 | 4 | 5 |              left | i | right  |  left = 3 | 4 | right = 5   |  left = -1 | 0 | right = 1
    //    +---+---+---+                   +---+        |           +---+             |           +---+
    //    | 6 | 7 | 8 |                   lower        |         lower = 7           |         lower = 3
    //    +---+---+---+
    // 
    //    
    //    +----------------------------------+
    //    | (rank_coords[0], rank_coords[1]) |
    //    +----------------------------------+
    // 
    //    +------------------+------------------+       +----------------------------+
    //    |      (0, 0)      |      (0, 1)      | * * * |      (0, dims[1] - 1)      |
    //    +------------------+------------------+       +----------------------------+
    //    |      (1, 0)      |      (1, 1)      | * * * |      (1, dims[1] - 1)      |
    //    +------------------+------------------+       +----------------------------+
    //                     * * *                    *                * * *
    //    +------------------+------------------+       +----------------------------+
    //    | (dims[0] - 1, 0) | (dims[0] - 1, 1) | * * * | (dims[0] - 1, dims[1] - 1) |
    //    +------------------+------------------+       +----------------------------+
    // 
    ///////////////////////////////////////////////////////

    // Определение процессов "слева" и  "справа"
    ierror = MPI_Cart_shift(communicator, 1, 1, &left_rank, &right_rank);
    if (ierror != MPI_SUCCESS)
        printf("MPI_Cart_shift error!");

    // Определение процессов "снизу" и  "сверху"
    ierror = MPI_Cart_shift(communicator, 0, 1, &upper_rank, &lower_rank);
    if (ierror != MPI_SUCCESS)
        printf("MPI_Cart_shift error!");
#pragma endregion

    // создание структуры Vars для MPI пересылок
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

    // Определение размеров подобластей для каждого процесса.
    // По направлению x
    int h1 = (I + 2 * PML_Size) / dims[0];
    int m1 = (I + 2 * PML_Size) % dims[0];

    // По направлению y
    int h2 = (I + 2 * PML_Size) / dims[1];
    int m2 = (I + 2 * PML_Size) % dims[1];

    // Учет нецелочисленного деления
    if (rank_coords[0] < m1)
        h1++;
    if (rank_coords[1] < m2)
        h2++;

    // Размер расчетной подобласти.
    // По направлениям x и y подобласть должна хранить
    // значения сеточных функций и параметры среды соседних подобластей для расчета по разностным схемам.
    _size_i = h1 + 4;
    _size_j = h2 + 4;
    _size_k = I + 2 * PML_Size + 4;

    // Случай dims[0] == 1 означает, что разбиение на подобласти по направлению x не производится
    if (dims[0] != 1)
    {
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
    }
    else
    {
        starti_rw = 2 + PML_Size;
        endi_rw = _size_i - PML_Size - 2;
    }

    // Случай dims[1] == 1 означает, что разбиение на подобласти по направлению y не производится
    if (dims[1] != 1)
    {
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
    }
    else
    {
        startj_rw = 2 + PML_Size;
        endj_rw = _size_j - PML_Size - 2;
    }

    // Размеры исходной вычислительной области и ее размеры в подобласти процесса.
    // В случае dims[0] == 1, endi_rw - starti_rw = I
    // В случае dims[1] == 1, endj_rw - startj_rw = I
    int gsizes[3] = { I , I , I};
    int lsizes[3] = { (endi_rw - starti_rw), (endj_rw - startj_rw), I};
    count_rw = lsizes[0] * lsizes[1] * lsizes[2];
    buf_rw = new double[count_rw];

    // Смещение по трем направлениям: x, y, z
    int starts[3];

    // Определение смещения для параллельного чтения / записи
    starts[0] = 0;
    if ((rank_coords[0] != 0) && (dims[0] != 1))
    {
        if (rank_coords[0] < m1)
        {
            starts[0] = rank_coords[0] * h1 - PML_Size;
        }
        else
        {
            starts[0] = m1 * (h1 + 1) + (rank_coords[0] - m1) * h1 - PML_Size;
        }
    }
    
    starts[1] = 0;
    if ((rank_coords[1] != 0) && (dims[1] != 1))
    {
        if (rank_coords[1] < m2)
        {
            starts[1] = rank_coords[1] * h2 - PML_Size;
        }
        else
        {
            starts[1] = m2 * (h2 + 1) + (rank_coords[1] - m2) * h2 - PML_Size;
        }
    }

    starts[2] = 0;

    starts_f[0] = starts[0];
    starts_f[1] = starts[1];
    starts_f[2] = starts[2];

    //printf("rank: (%d, %d)\n    starts i = %d\n    starts j = %d\n    c rw = %d\n    l i = %d\n    l j = %d\n", rank_coords[0], rank_coords[1], starts[0], starts[1], count_rw, lsizes[0], lsizes[1]);

    // Определяем подмассив - подобласть в трехмерном пространстве.
    // subarray_rw - определяет место положение данных в файле для считывания / записи данному процессу.
    MPI_Type_create_subarray(3, gsizes, lsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &subarray_rw);
    MPI_Type_commit(&subarray_rw);

    gsizes[0] = _size_i;
    gsizes[1] = _size_j;
    gsizes[2] = _size_k;

    // Левая граница для отправки
    lsizes[0] = _size_i - 4;
    lsizes[1] = 2;
    lsizes[2] = _size_k - 4;

    starts[0] = 2;
    starts[1] = 2;
    starts[2] = 2;
    MPI_Type_create_subarray(3, gsizes, lsizes, starts, MPI_ORDER_C, mpi_vars, &left_subaray_send);
    MPI_Type_commit(&left_subaray_send);

    // Левая граница для чтения
    starts[1] = 0;
    MPI_Type_create_subarray(3, gsizes, lsizes, starts, MPI_ORDER_C, mpi_vars, &left_subaray_recv);
    MPI_Type_commit(&left_subaray_recv);

    // правая граница для отправки
    starts[1] = _size_j - 4;
    MPI_Type_create_subarray(3, gsizes, lsizes, starts, MPI_ORDER_C, mpi_vars, &right_subaray_send);
    MPI_Type_commit(&right_subaray_send);

    // правая граница для чтения
    starts[1] = _size_j - 2;
    MPI_Type_create_subarray(3, gsizes, lsizes, starts, MPI_ORDER_C, mpi_vars, &right_subaray_recv);
    MPI_Type_commit(&right_subaray_recv);


    // верхняя граница для отправки
    lsizes[0] = 2;
    lsizes[1] = _size_j - 4;
    lsizes[2] = _size_k - 4;

    starts[0] = 2;
    starts[1] = 2;
    starts[2] = 2;
    MPI_Type_create_subarray(3, gsizes, lsizes, starts, MPI_ORDER_C, mpi_vars, &upper_subaray_send);
    MPI_Type_commit(&upper_subaray_send);

    // верхняя граница для чтения
    starts[0] = 0;
    MPI_Type_create_subarray(3, gsizes, lsizes, starts, MPI_ORDER_C, mpi_vars, &upper_subaray_recv);
    MPI_Type_commit(&upper_subaray_recv);

    // нижняя граница для отправки
    starts[0] = _size_i - 4;
    MPI_Type_create_subarray(3, gsizes, lsizes, starts, MPI_ORDER_C, mpi_vars, &lower_subaray_send);
    MPI_Type_commit(&lower_subaray_send);

    // нижняя граница для чтения
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

    // определение максимального значение скорости продольных волн для задания шага по времени
#pragma region Find max vp
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
#pragma endregion

    this->tau = (h / max * 0.95);
    this->N = int(T / (tau * 3));

    // Определение количества файлов и количества итераций в файле.
    // Каждый файл содержит в себе один из расчетных параметров
    long int size_data_one_iter = I * I * I * 8; // в байтах
    long int size_data = N * size_data_one_iter; // в байтах
    this->num_iters_in_file = (int)(this->max_size_file / size_data_one_iter);
    this->num_file = (int)(N / this->num_iters_in_file) + 1;

    w_rank_curr = new Vars[_size_i * _size_j * _size_k];
    w_rank_next = new Vars[_size_i * _size_j * _size_k];
    g_rank_next = new Vars[_size_i * _size_j * _size_k];
    g_rank_curr = new Vars[_size_i * _size_j * _size_k];
    w_next = new Vars[_size_i * _size_j * _size_k];
    w_curr = new Vars[_size_i * _size_j * _size_k];

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
                w_curr[index] = w0;
                w_next[index] = w0;
                ++index;
            }
        }
    }
}

Wave3d::~Wave3d()
{
    delete[] w_rank_curr;
    delete[] w_rank_next;
    delete[] g_rank_curr;
    delete[] g_rank_next;
    delete[] w_curr;
    delete[] w_next;
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

    _write_to_file_Uz();
    _write_to_file_Abs_U();

    long int index;
#pragma omp parallel default(shared) num_threads(4)
    for (int n = 0; n < N; n++)
    {
#pragma omp for private(index)
        for (int i = 2; i < _size_i - 2; ++i)
        {
            for (int j = 2; j < _size_j - 2; ++j)
            {
                for (int k = 2; k < _size_k - 2; ++k)
                {
                    index = i * _I + j * _J + k;
                    w_rank_curr[index] = w_curr[index];
                }
            }
        }

        _make_step_X();
        _make_step_Y();
        _make_step_Z();

#pragma omp for private(index)
        for (int i = 2; i < _size_i - 2; ++i)
        {
            for (int j = 2; j < _size_j - 2; ++j)
            {
                for (int k = 2; k < _size_k - 2; ++k)
                {
                    index = i * _I + j * _J + k;
                    w_next[index] = w_rank_curr[index] / 6.;
                    w_rank_curr[index] = w_curr[index];
                }
            }
        }

        _make_step_Y();
        _make_step_X();
        _make_step_Z();

#pragma omp for private(index)
        for (int i = 2; i < _size_i - 2; ++i)
        {
            for (int j = 2; j < _size_j - 2; ++j)
            {
                for (int k = 2; k < _size_k - 2; ++k)
                {
                    index = i * _I + j * _J + k;
                    w_next[index] += w_rank_curr[index] / 6.;
                    w_rank_curr[index] = w_curr[index];
                }
            }
        }

        _make_step_Z();
        _make_step_X();
        _make_step_Y();

#pragma omp for private(index)
        for (int i = 2; i < _size_i - 2; ++i)
        {
            for (int j = 2; j < _size_j - 2; ++j)
            {
                for (int k = 2; k < _size_k - 2; ++k)
                {
                    index = i * _I + j * _J + k;
                    w_next[index] += w_rank_curr[index] / 6.;
                    w_rank_curr[index] = w_curr[index];
                }
            }
        }

        _make_step_X();
        _make_step_Z();
        _make_step_Y();

#pragma omp for private(index)
        for (int i = 2; i < _size_i - 2; ++i)
        {
            for (int j = 2; j < _size_j - 2; ++j)
            {
                for (int k = 2; k < _size_k - 2; ++k)
                {
                    index = i * _I + j * _J + k;
                    w_next[index] += w_rank_curr[index] / 6.;
                    w_rank_curr[index] = w_curr[index];
                }
            }
        }

        _make_step_Y();
        _make_step_Z();
        _make_step_X();

#pragma omp for private(index)
        for (int i = 2; i < _size_i - 2; ++i)
        {
            for (int j = 2; j < _size_j - 2; ++j)
            {
                for (int k = 2; k < _size_k - 2; ++k)
                {
                    index = i * _I + j * _J + k;
                    w_next[index] += w_rank_curr[index] / 6.;
                    w_rank_curr[index] = w_curr[index];
                }
            }
        }

        _make_step_Z();
        _make_step_Y();
        _make_step_X();

#pragma omp for private(index)
        for (int i = 2; i < _size_i - 2; ++i)
        {
            for (int j = 2; j < _size_j - 2; ++j)
            {
                for (int k = 2; k < _size_k - 2; ++k)
                {
                    index = i * _I + j * _J + k;
                    w_curr[index] = w_next[index] + w_rank_curr[index] / 6.;
                }
            }
        }

        F(n);
        _write_to_file_Abs_U();
        _write_to_file_Uz();
        //std::cout << "iter " << n << "complete\n";
    }
}