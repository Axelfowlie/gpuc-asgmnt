

int prefix(const __global uint* mortoncodes, int a, int b, int N) {

  // If b is out of range return -1 for the prefix length
  if (b < 0 || b >= N) return -1;

  uint ca = mortoncodes[a];
  uint cb = mortoncodes[b];

  return clz(ca ^ cb);
}


uint2 determineRange(const __global uint* mortoncodes, int GID, int N) {

  // sign - apparently no overload for int...
  int d = sign((float)(prefix(mortoncodes, GID, GID + 1, N) -  prefix(mortoncodes, GID, GID - 1, N)));

  int minPrefix = prefix(mortoncodes, GID, GID - d, N);

  // Compute the upper bound for the other end of the range (lower bound for negative d, but we will use upper bound only...)
  int upperbound = 2;
  while (prefix(mortoncodes, GID, GID + upperbound * d, N) > minPrefix) upperbound <<= 1;

  // Start searching for the other end of the range from upperbound
  int stride = upperbound >> 1;
  int l = 0;
  while (stride != 0) {
    if (prefix(mortoncodes, GID, GID + (l + stride) * d, N) > minPrefix) l += stride;
    stride >>= 1;
  }

  int other = GID + l * d;
  if (other < GID) return (uint2)(other, GID);
  else return (uint2)(GID, other);
}


int findSplit(const __global uint* mortoncodes, int first, int last) {

  uint firstCode = mortoncodes[first];
  uint lastCode = mortoncodes[last];

  // Split range of identical codes in the middle
  if (firstCode == lastCode) return (first + last) >> 1;

  // count leading zeros of XOR of the two codes, corresponds to number of leading bits that are not the same 
  int referencePrefix = clz(firstCode ^ lastCode);

  // Binary search for the highest code, that has a longer common prefix than the reference prefix
  // Start with the first element in the range and stride of the whole range
  int split = first;
  int step = last - first;

  do {
    // decrease step size exponentially, and calculate the potential split position
    step = (step + 1) >> 1;
    int newSplit = split + step;

    if (newSplit < last) {
      uint splitCode = mortoncodes[newSplit];
      int splitPrefix = clz(firstCode ^ splitCode);
      // A larger prefix than referencePrefix means, that we still haven't advanced far enough towards the right 
      if (splitPrefix > referencePrefix) split = newSplit;
    }
  } while (step > 1);

  return split;
}


__kernel void NodesHierarchy(__global uint2* childnodes, __global uint* parents, const __global uint* mortoncodes, uint N) {
  int GID = get_global_id(0);

  if (GID >= N - 1) return;

  // Determine the node range and use it to find the split position
  uint2 range = determineRange(mortoncodes, GID, N);
  int split = findSplit(mortoncodes, range.x, range.y);

  // left and right child are split and split + 1
  // If the child is the same as the left/right end of the range, the subrange [range.x, leftChild] or [rightChild, range.y] respectively consists only of a single node
  int leftChild = split;
  if (leftChild == range.x) leftChild += N;

  int rightChild = split + 1;
  if (rightChild == range.y) rightChild += N;

  // Set the child nodes in this internal node
  childnodes[GID] = (uint2)(leftChild, rightChild);
  // Set the index of the parent in the two child nodes
  parents[leftChild] = GID;
  parents[rightChild] = GID;
}
