#include "cluster.h"
#include "cluster_cl.h"
#include "ocl_macros.h"

void allocateSSTData_cl(int max_strips, cl_context context, sst_data_cl_t* sst_data_cl) {
  cl_int clStatus;

  sst_data_cl->detId = clCreateBuffer(context, CL_MEM_READ_ONLY, max_strips*sizeof(detId_t), NULL, &clStatus);
  sst_data_cl->stripId = clCreateBuffer(context, CL_MEM_READ_ONLY, max_strips*sizeof(uint16_t), NULL, &clStatus);
  sst_data_cl->adc = clCreateBuffer(context, CL_MEM_READ_ONLY, max_strips*sizeof(uint8_t), NULL, &clStatus);
  sst_data_cl->fedId = clCreateBuffer(context, CL_MEM_READ_ONLY, max_strips*sizeof(fedId_t), NULL, &clStatus);
  sst_data_cl->fedCh = clCreateBuffer(context, CL_MEM_READ_ONLY, max_strips*sizeof(fedCh_t), NULL, &clStatus);
  sst_data_cl->seedStripsNCIndex = clCreateBuffer(context, CL_MEM_READ_ONLY, max_strips*sizeof(int), NULL, &clStatus);
}

void allocateCalibData_cl(int max_strips, cl_context context, calib_data_cl_t* calib_data_cl) {
  cl_int clStatus;

  calib_data_cl->noise = clCreateBuffer(context, CL_MEM_READ_ONLY, max_strips*sizeof(float), NULL, &clStatus);
  calib_data_cl->gain = clCreateBuffer(context, CL_MEM_READ_ONLY, max_strips*sizeof(float), NULL, &clStatus);
  calib_data_cl->bad = clCreateBuffer(context, CL_MEM_READ_ONLY, max_strips*sizeof(bool), NULL, &clStatus);
}

void allocateClustData_cl(int max_seedstrips, cl_context context, clust_data_cl_t* clust_data_cl) {
  cl_int clStatus;
  
  clust_data_cl->clusterLastIndexLeft = clCreateBuffer(context, CL_MEM_WRITE_ONLY, max_seedstrips*sizeof(int), NULL, &clStatus);
  clust_data_cl->clusterLastIndexRight = clCreateBuffer(context, CL_MEM_WRITE_ONLY, max_seedstrips*sizeof(int), NULL, &clStatus);
  clust_data_cl->clusterADCs = clCreateBuffer(context, CL_MEM_WRITE_ONLY, max_seedstrips*256*sizeof(uint8_t), NULL, &clStatus);
  clust_data_cl->trueCluster = clCreateBuffer(context, CL_MEM_WRITE_ONLY, max_seedstrips*sizeof(bool), NULL, &clStatus);
  clust_data_cl->barycenter = clCreateBuffer(context, CL_MEM_WRITE_ONLY, max_seedstrips*sizeof(float), NULL, &clStatus);
}

void freeSSTData_cl(sst_data_cl_t *sst_data_cl) {
  cl_int clStatus;

  clStatus = clReleaseMemObject(sst_data_cl->detId);
  clStatus = clReleaseMemObject(sst_data_cl->stripId);
  clStatus = clReleaseMemObject(sst_data_cl->adc);
  clStatus = clReleaseMemObject(sst_data_cl->fedId);
  clStatus = clReleaseMemObject(sst_data_cl->fedCh);
  clStatus = clReleaseMemObject(sst_data_cl->seedStripsNCIndex);
}

void freeCalibData_cl(calib_data_cl_t *calib_data_cl) {
  cl_int clStatus;

  clStatus = clReleaseMemObject(calib_data_cl->noise);
  clStatus = clReleaseMemObject(calib_data_cl->gain);
  clStatus = clReleaseMemObject(calib_data_cl->bad);
}

void freeClustData_cl(clust_data_cl_t *clust_data_cl) {
  cl_int clStatus;

  clStatus = clReleaseMemObject(clust_data_cl->clusterLastIndexLeft);
  clStatus = clReleaseMemObject(clust_data_cl->clusterLastIndexRight);
  clStatus = clReleaseMemObject(clust_data_cl->clusterADCs);
  clStatus = clReleaseMemObject(clust_data_cl->trueCluster);
  clStatus = clReleaseMemObject(clust_data_cl->barycenter);

}

void copyDataToFPGA(cl_command_queue command_queue, sst_data_t* sst_data, sst_data_cl_t* sst_data_cl, calib_data_t* calib_data, calib_data_cl_t* calib_data_cl) {
  int nStrips =sst_data->nStrips;
  sst_data_cl->nStrips = nStrips;
  sst_data_cl->nSeedStripsNC = sst_data->nSeedStripsNC;

  cl_int clStatus;

  clStatus = clEnqueueWriteBuffer(command_queue, sst_data_cl->detId, CL_TRUE, 0, nStrips*sizeof(detId_t), sst_data->detId, 0, NULL, NULL);
  clStatus = clEnqueueWriteBuffer(command_queue, sst_data_cl->stripId, CL_TRUE, 0, nStrips*sizeof(uint16_t), sst_data->stripId, 0, NULL, NULL);
  clStatus = clEnqueueWriteBuffer(command_queue, sst_data_cl->adc, CL_TRUE, 0, nStrips*sizeof(uint8_t), sst_data->adc, 0, NULL, NULL);
  clStatus = clEnqueueWriteBuffer(command_queue, sst_data_cl->fedId, CL_TRUE, 0, nStrips*sizeof(fedId_t), sst_data->fedId, 0, NULL, NULL);
  clStatus = clEnqueueWriteBuffer(command_queue, sst_data_cl->fedCh, CL_TRUE, 0, nStrips*sizeof(fedCh_t), sst_data->fedCh, 0, NULL, NULL);
  clStatus = clEnqueueWriteBuffer(command_queue, sst_data_cl->seedStripsNCIndex, CL_TRUE, 0, nStrips*sizeof(int), sst_data->seedStripsNCIndex, 0, NULL, NULL);

  clStatus = clEnqueueWriteBuffer(command_queue, calib_data_cl->noise, CL_TRUE, 0, nStrips*sizeof(float), calib_data->noise, 0, NULL, NULL);
  clStatus = clEnqueueWriteBuffer(command_queue, calib_data_cl->gain, CL_TRUE, 0, nStrips*sizeof(float), calib_data->gain, 0, NULL, NULL);
  clStatus = clEnqueueWriteBuffer(command_queue, calib_data_cl->bad, CL_TRUE, 0, nStrips*sizeof(bool), calib_data->bad, 0, NULL, NULL);
}

void findCluster(cl_program program, cl_command_queue command_queue, sst_data_cl_t* sst_data_cl, calib_data_cl_t* calib_data_cl, clust_data_cl_t* clust_data_cl) {
  cl_int clStatus;

  const char *kernel_name1="findLeftRightBoundary";
  cl_kernel findBoundary_kernel = clCreateKernel(program, kernel_name1, &clStatus);
  LOG_OCL_ERROR(clStatus, "Failed to create findLeftRightBoundary kernel");

  clStatus = clSetKernelArg(findBoundary_kernel, 0, sizeof(int),  (void *)&sst_data_cl->nStrips);
  clStatus = clSetKernelArg(findBoundary_kernel, 1, sizeof(cl_mem), (void *)&sst_data_cl->detId);
  clStatus = clSetKernelArg(findBoundary_kernel, 2, sizeof(cl_mem), (void *)&sst_data_cl->stripId);
  clStatus = clSetKernelArg(findBoundary_kernel, 3, sizeof(cl_mem), (void *)&sst_data_cl->adc);
  clStatus = clSetKernelArg(findBoundary_kernel, 4, sizeof(cl_mem), (void *)&sst_data_cl->seedStripsNCIndex);
  clStatus = clSetKernelArg(findBoundary_kernel, 5, sizeof(cl_mem), (void *)&calib_data_cl->noise);
  clStatus = clSetKernelArg(findBoundary_kernel, 6, sizeof(cl_mem), (void *)&clust_data_cl->clusterLastIndexLeft);
  clStatus = clSetKernelArg(findBoundary_kernel, 7, sizeof(cl_mem), (void *)&clust_data_cl->clusterLastIndexRight);
  clStatus = clSetKernelArg(findBoundary_kernel, 8, sizeof(cl_mem), (void *)&clust_data_cl->trueCluster);
  LOG_OCL_ERROR(clStatus, "Failed to set kernel arguments for findboundary_kernel" );

  const char *kernel_name2="checkClusterCondition";
  cl_kernel checkCondition_kernel = clCreateKernel(program, kernel_name2, &clStatus);
  LOG_OCL_ERROR(clStatus, "Failed to create checkClusterCondition kernel");

  clStatus = clSetKernelArg(checkCondition_kernel, 0, sizeof(int),  (void *)&sst_data_cl->nSeedStripsNC);
  clStatus = clSetKernelArg(checkCondition_kernel, 1, sizeof(cl_mem), (void *)&sst_data_cl->stripId);
  clStatus = clSetKernelArg(checkCondition_kernel, 2, sizeof(cl_mem), (void *)&sst_data_cl->adc);
  clStatus = clSetKernelArg(checkCondition_kernel, 3, sizeof(cl_mem),  (void *)&calib_data_cl->gain);
  clStatus = clSetKernelArg(checkCondition_kernel, 4, sizeof(cl_mem), (void *)&clust_data_cl->clusterLastIndexLeft);
  clStatus = clSetKernelArg(checkCondition_kernel, 5, sizeof(cl_mem), (void *)&clust_data_cl->clusterLastIndexRight);
  clStatus = clSetKernelArg(checkCondition_kernel, 6, sizeof(cl_mem), (void *)&clust_data_cl->trueCluster);
  clStatus = clSetKernelArg(checkCondition_kernel, 7, sizeof(cl_mem), (void *)&clust_data_cl->clusterADCs);
  clStatus = clSetKernelArg(checkCondition_kernel, 8, sizeof(cl_mem), (void *)&clust_data_cl->barycenter);
  LOG_OCL_ERROR(clStatus, "Failed to set kernel arguments for checkCondition_kernel" );

  size_t global_size = sst_data_cl->nSeedStripsNC;

  clStatus = clEnqueueNDRangeKernel(command_queue, findBoundary_kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
  clStatus = clEnqueueNDRangeKernel(command_queue, checkCondition_kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);

  // Clean up and wait for all the comands to complete.
  clStatus = clFinish(command_queue);
  clStatus = clReleaseKernel(findBoundary_kernel);
  clStatus = clReleaseKernel(checkCondition_kernel);

}

void copyDataBackCPU(cl_command_queue command_queue, clust_data_t* clust_data, clust_data_cl_t* clust_data_cl){

  int nSeedStripsNC=150000;
  
  cl_int clStatus;
  clStatus = clEnqueueReadBuffer(command_queue, clust_data_cl->clusterLastIndexLeft, CL_TRUE, 0, nSeedStripsNC*sizeof(int), clust_data->clusterLastIndexLeft, 0, NULL, NULL);
  clStatus = clEnqueueReadBuffer(command_queue,clust_data_cl->clusterLastIndexRight, CL_TRUE, 0, nSeedStripsNC*sizeof(int), clust_data->clusterLastIndexRight, 0, NULL, NULL);
  clStatus = clEnqueueReadBuffer(command_queue,clust_data_cl->clusterADCs, CL_TRUE, 0, nSeedStripsNC*256*sizeof(uint8_t), clust_data->clusterADCs, 0, NULL, NULL);
  clStatus = clEnqueueReadBuffer(command_queue,clust_data_cl->trueCluster, CL_TRUE, 0, nSeedStripsNC*sizeof(bool), clust_data->trueCluster, 0, NULL, NULL);
  clStatus = clEnqueueReadBuffer(command_queue,clust_data_cl->barycenter, CL_TRUE, 0, nSeedStripsNC*sizeof(float), clust_data->barycenter, 0, NULL, NULL);

}
