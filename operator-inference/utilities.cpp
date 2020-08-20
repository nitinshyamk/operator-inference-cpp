#include "utilities.h"

void print_mat(const double* arr, int M, int N)
{
	using std::cout;
	using std::endl;
	if (M * N > 1000)
	{
		cout << "Matrix is too large to print." << endl;
		return;
	}

	cout << "Printing matrix of dimensions " << M << " x " << N << endl;
	for (int r = 0; r < M; ++r)
	{
		for (int c = 0; c < N; ++c)
		{
			cout << arr[columnMajorZeroIndex(r, c, M, N)] << '\t';
		}
		cout << endl;
	}
}