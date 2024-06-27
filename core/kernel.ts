type KernelFunction = (x: number, y: number, bandwidth: number) => number;

function NadarayaWatson(
  X: number[],
  Y: number[],
  Z: number,
  bandwidth: number,
  kernel: KernelFunction = epanechnikovKernel
): number {
  const weight1: number = X.reduce((sum, x) => sum + kernel(x, Z, bandwidth), 0);
  const weight2: number = X.reduce(
    (sum, x, index) => sum + kernel(x, Z, bandwidth) * Math.pow(Y[index], 2),
    0
  );

  return weight2 / weight1;
}

function epanechnikovKernel(x: number, y: number, bandwidth: number): number {
  const u = Math.abs((x - y) / bandwidth);
  if (u <= 1) {
    return 0.75 * (1 - u * u);
  }
  return 0;
}

// Example usage:
const X: number[] = [1, 2, 3, 4];
const Y: number[] = [2, 3, 4, 5];
const Z: number = 3;
const bandwidth: number = 0.5;

const result: number = NadarayaWatson(X, Y, Z, bandwidth);
console.log(result);

