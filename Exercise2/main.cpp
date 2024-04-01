#include <iostream>
#include <iomanip>
#include "Eigen/Eigen"

using namespace std;
using namespace Eigen;

void relativeError(const VectorXd& x, const VectorXd& exactSolution)
{
    double relative_error = (x - exactSolution).norm() / exactSolution.norm();
    cout << "Relative Error: " << scientific << setprecision(16) << relative_error << endl;
}

int main()
{
    // System 1
    Matrix2d A1;
    A1 << 5.547001962252291e-01, -3.770900990025203e-02,
          8.320502943378437e-01, -9.992887623566787e-01;
    Vector2d b1;
    b1 << -5.169911863249772e-01, 1.672384680188350e-01;
    Vector2d exactSolution1;
    exactSolution1 << -1.0e+00, -1.0e+00;

    // PALU decomposition
    VectorXd x_palu1 = A1.fullPivLu().solve(b1);
    cout << "System 1: PALU Solution:\n" << scientific << setprecision(1) << x_palu1 << endl;
    relativeError(x_palu1, exactSolution1);

    // QR decomposition
    VectorXd x_qr1 = A1.householderQr().solve(b1);
    cout << "System 1: QR Solution:\n" << scientific << setprecision(1) << x_qr1 << endl;
    relativeError(x_qr1, exactSolution1);

    // System 2
    Matrix2d A2;
    A2 << 5.547001962252291e-01, -5.540607316466765e-01,
          8.320502943378437e-01, -8.324762492991313e-01;
    Vector2d b2;
    b2 << -6.394645785530173e-04, 4.259549612877223e-04;
    Vector2d exactSolution2;
    exactSolution2 << -1.0e+00, -1.0e+00;

    // PALU decomposition
    VectorXd x_palu2 = A2.fullPivLu().solve(b2);
    cout << "\nSystem 2: PALU Solution:\n" << scientific << setprecision(1) << x_palu2 << endl;
    relativeError(x_palu2, exactSolution2);

    // QR decomposition
    VectorXd x_qr2 = A2.householderQr().solve(b2);
    cout << "System 2: QR Solution:\n" << scientific << setprecision(1) << x_qr2 << endl;
    relativeError(x_qr2, exactSolution2);

    // System 3
    Matrix2d A3;
    A3 << 5.547001962252291e-01, -5.547001955851905e-01,
          8.320502943378437e-01, -8.320502947645361e-01;
    Vector2d b3;
    b3 << -6.400391328043042e-10, 4.266924591433963e-10;
    Vector2d exactSolution3;
    exactSolution3 << -1.0e+00, -1.0e+00;

    // PALU decomposition
    VectorXd x_palu3 = A3.fullPivLu().solve(b3);
    cout << "\nSystem 3: PALU Solution:\n" << scientific << setprecision(1) << x_palu3 << endl;
    relativeError(x_palu3, exactSolution3);

    // QR decomposition
    VectorXd x_qr3 = A3.householderQr().solve(b3);
    cout << "System 3: QR Solution:\n" << scientific << setprecision(1) << x_qr3 << endl;
    relativeError(x_qr3, exactSolution3);

    return 0;
}
