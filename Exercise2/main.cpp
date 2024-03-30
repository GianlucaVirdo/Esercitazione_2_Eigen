#include <iostream>
#include "Eigen/Eigen"

using namespace std;
using namespace Eigen;


VectorXd SolveSystemPALU(const MatrixXd& A, const VectorXd& b)
{
    unsigned int n = A.rows();
    VectorXd x(n);

    x = A.fullPivLu().solve(b);

    return x;
}

VectorXd SolveSystemQR(const MatrixXd& A, const VectorXd& b)
{
    int n = A.rows();
    VectorXd x(n);

    x = A.fullPivHouseholderQr().solve(b);

    return x;
}

void CheckSolution(const MatrixXd& A, const VectorXd& b, const VectorXd& trueSolution, double& errRelPALU, double& errRelQR)
{
    VectorXd xPALU = SolveSystemPALU(A, b);
    VectorXd xQR = SolveSystemQR(A, b);

    errRelPALU = (xPALU - trueSolution).norm() / trueSolution.norm();
    errRelQR = (xQR - trueSolution).norm() / trueSolution.norm();
}


int main()
{
    Vector2d trueSolution;
    trueSolution << -1.0e+0, -1.0e+0;

    Matrix2d A1;
    A1 << 5.547001962252291e-01, -3.770900990025203e-02, 8.320502943378437e-01, -9.992887623566787e-01;

    Vector2d b1;
    b1 << -5.169911863249772e-01, 1.672384680188350e-01;

    double errRel1PALU;
    double errRel1QR;

    CheckSolution(A1, b1, trueSolution, errRel1PALU, errRel1QR);

    cout << "Relative errors:" << endl;
    cout << scientific << "First matrix - " << "PALU: " << errRel1PALU << " QR: " << errRel1QR << endl;

    Matrix2d A2;
    A2 << 5.547001962252291e-01, -5.540607316466765e-01, 8.320502943378437e-01, -8.324762492991313e-01;

    Vector2d b2;
    b2 << -6.394645785530173e-04, 4.259549612877223e-04;

    double errRel2PALU;
    double errRel2QR;

    CheckSolution(A2, b2, trueSolution, errRel2PALU, errRel2QR);

    cout << scientific << "Second matrix - " << "PALU: " << errRel2PALU << " QR: " << errRel2QR << endl;

    Matrix2d A3;
    A3 << 5.547001962252291e-01, -5.547001955851905e-01, 8.320502943378437e-01, -8.320502947645361e-01;

    Vector2d b3;
    b3 << -6.400391328043042e-10, 4.266924591433963e-10;

    double errRel3PALU;
    double errRel3QR;

    CheckSolution(A3, b3, trueSolution, errRel3PALU, errRel3QR);

    cout << scientific << "Third matrix - " << "PALU: " << errRel3PALU << " QR: " << errRel3QR << endl;

    return 0;
}
