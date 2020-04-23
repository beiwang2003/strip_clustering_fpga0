#include <fstream>
#include <iostream>
#include <cstring>
#include <memory>
#include <functional>
#include "cluster.h"
#ifdef USE_FPGA
#include <CL/cl.h>
#include "cluster_cl.h"
#include "ocl_macros.h"
#include "opencl.h"
#define STRING_BUFFER_LEN 1024

static void display_device_info( cl_device_id device );
#endif 

int main()
{
  const int max_strips = MAX_STRIPS;
  const int max_seedstrips = MAX_SEEDSTRIPS;
  auto sst_data_ptr = std::make_unique<sst_data_t>();
  auto clust_data_ptr = std::make_unique<clust_data_t>();
  auto calib_data_ptr = std::make_unique<calib_data_t>();

#ifdef USE_FPGA
  cl_int clStatus;

  // Get platform and device information                                                                               
  cl_platform_id * platforms = NULL;
  // Set up the Platform. Take a look at the MACROs used in this file.                                                 
  // These are defined in common/ocl_macros.h                                                                          
  OCL_CREATE_PLATFORMS( platforms );
  // Print out the Plaftform info
  {
    char char_buffer[STRING_BUFFER_LEN];
    printf("Querying platform for info:\n");
    printf("==========================\n");
    clGetPlatformInfo(platforms[0], CL_PLATFORM_NAME, STRING_BUFFER_LEN, char_buffer, NULL);
    printf("%-40s = %s\n", "CL_PLATFORM_NAME", char_buffer);
    clGetPlatformInfo(platforms[0], CL_PLATFORM_VENDOR, STRING_BUFFER_LEN, char_buffer, NULL);
    printf("%-40s = %s\n", "CL_PLATFORM_VENDOR ", char_buffer);
    clGetPlatformInfo(platforms[0], CL_PLATFORM_VERSION, STRING_BUFFER_LEN, char_buffer, NULL);
    printf("%-40s = %s\n\n", "CL_PLATFORM_VERSION ", char_buffer);
  }

  // Get the devices list and choose the type of device you want to run on                                             
  cl_device_id *device_list = NULL;
  OCL_CREATE_DEVICE( platforms[0], CL_DEVICE_TYPE_ALL, device_list);
  display_device_info(device_list[0]);

  // Create OpenCL context for devices in device_list                                                                  
  cl_context context;
  cl_context_properties props[3] =
    {
      CL_CONTEXT_PLATFORM,
      (cl_context_properties)platforms[0],
      0
    };

  // An OpenCL context can be associated to multiple devices, either CPU or GPU                                        
  // based on the value of DEVICE_TYPE defined above. Here we use only 1 device                                                                  
  context = clCreateContext( NULL, 1, &device_list[0], NULL, NULL, &clStatus);
  LOG_OCL_ERROR(clStatus, "clCreateContext Failed..." );
  
  // Create a command queue for the first device in device_list                                                        
  cl_command_queue command_queue = clCreateCommandQueue(context, device_list[0], 0, &clStatus);
  LOG_OCL_ERROR(clStatus, "clCreateCommandQueue Failed..." );

  // Create the program.   
  std::string binary_file = aocl_utils::getBoardBinaryFile("clustering", device_list[0]);
  printf("Using AOCX: %s\n", binary_file.c_str());
  cl_program program = aocl_utils::createProgramFromBinary(context, binary_file.c_str(), &device_list[0], 1);
  LOG_OCL_ERROR(clStatus, "clCreateProgramWithSource Failed..." );

  // Build the program that was just created.
  clStatus = clBuildProgram(program, 0, NULL, "", NULL, NULL);
  LOG_OCL_ERROR(clStatus, "Failed to build program...");
#endif 

  allocateSSTData(max_strips, sst_data_ptr.get());
  allocateClustData(max_strips, clust_data_ptr.get());
  allocateCalibData(max_strips, calib_data_ptr.get());
 

#ifdef USE_FPGA
  auto sst_data_cl_ptr = std::make_unique<sst_data_cl_t>();
  auto clust_data_cl_ptr = std::make_unique<clust_data_cl_t>();
  auto calib_data_cl_ptr = std::make_unique<calib_data_cl_t>();

  allocateSSTData_cl(max_strips, context, sst_data_cl_ptr.get());
  allocateCalibData_cl(max_strips, context, calib_data_cl_ptr.get());
  allocateClustData_cl(max_seedstrips, context, clust_data_cl_ptr.get());
#endif

  // read in calibration data
  std::string condfilename("stripcond.bin");
  auto conditions = std::make_unique<SiStripConditions>(condfilename);

  std::vector<FEDRawData> fedRawData;
  std::vector<FEDBuffer> fedBuffer;
  std::vector<fedId_t> fedIndex;
  FEDReadoutMode mode;

  std::string datafilename("stripdata.bin");
  readinRawData(datafilename, conditions.get(), fedRawData, fedBuffer, fedIndex, mode, sst_data_ptr.get());

  unpackRawData(conditions.get(), fedRawData, fedBuffer, fedIndex, sst_data_ptr.get(), calib_data_ptr.get(), mode);

  setSeedStripsNCIndex(sst_data_ptr.get(), calib_data_ptr.get(), conditions.get());

#ifdef USE_FPGA

  copyDataToFPGA(command_queue, sst_data_ptr.get(), sst_data_cl_ptr.get(), calib_data_ptr.get(), calib_data_cl_ptr.get());

  findCluster(program, command_queue, sst_data_cl_ptr.get(), calib_data_cl_ptr.get(), clust_data_cl_ptr.get());

  copyDataBackCPU(command_queue, clust_data_ptr.get(), clust_data_cl_ptr.get());

#else
  findCluster(sst_data_ptr.get(), calib_data_ptr.get(), conditions.get(), clust_data_ptr.get());
#endif

  // print out the result                                                                                                             
#ifdef OUTPUT
  std::cout<<" nSeedStripsNC "<<sst_data_ptr.get()->nSeedStripsNC<<std::endl;
  for (int j=0; j<sst_data_ptr.get()->nSeedStripsNC; j++) {
    if (clust_data_ptr.get()->trueCluster[j]){
      int index = clust_data_ptr.get()->clusterLastIndexLeft[j];
        //      std::cout<<" det id "<<sst_data[i]->detId[index]<<" strip "<<sst_data[i]->stripId[index]                              
      std::cout<<" bary center "<<clust_data_ptr.get()->barycenter[j]<<": ";
      int right=clust_data_ptr.get()->clusterLastIndexRight[j];
      int size=right-index+1;
      for (int k=0; k<size; k++){
	std::cout<<(unsigned int)clust_data_ptr.get()->clusterADCs[k*sst_data_ptr.get()->nSeedStripsNC+j]<<" ";
      }
      std::cout<<std::endl;
    }
  }
#endif

  freeSSTData(sst_data_ptr.get());   
  freeClustData(clust_data_ptr.get());
  freeCalibData(calib_data_ptr.get());

#ifdef USE_FPGA
  freeSSTData_cl(sst_data_cl_ptr.get());
  freeClustData_cl(clust_data_cl_ptr.get());
  freeCalibData_cl(calib_data_cl_ptr.get());

  clStatus = clReleaseProgram(program);
  clStatus = clReleaseCommandQueue(command_queue);
  clStatus = clReleaseContext(context);

  free(platforms);
  free(device_list);
#endif

  return 0;

}

#ifdef USE_FPGA
// Helper functions to display parameters returned by OpenCL queries                                                                  
static void device_info_ulong( cl_device_id device, cl_device_info param, const char* name) {
  cl_ulong a;
  clGetDeviceInfo(device, param, sizeof(cl_ulong), &a, NULL);
  printf("%-40s = %lu\n", name, a);
}
static void device_info_uint( cl_device_id device, cl_device_info param, const char* name) {
  cl_uint a;
  clGetDeviceInfo(device, param, sizeof(cl_uint), &a, NULL);
  printf("%-40s = %u\n", name, a);
}
static void device_info_bool( cl_device_id device, cl_device_info param, const char* name) {
  cl_bool a;
  clGetDeviceInfo(device, param, sizeof(cl_bool), &a, NULL);
  printf("%-40s = %s\n", name, (a?"true":"false"));
}
static void device_info_string( cl_device_id device, cl_device_info param, const char* name) {
  char a[STRING_BUFFER_LEN];
  clGetDeviceInfo(device, param, STRING_BUFFER_LEN, &a, NULL);
  printf("%-40s = %s\n", name, a);
}
static void display_device_info( cl_device_id device ) {

  printf("Querying device for info:\n");
  printf("========================\n");
  device_info_string(device, CL_DEVICE_NAME, "CL_DEVICE_NAME");
  device_info_string(device, CL_DEVICE_VENDOR, "CL_DEVICE_VENDOR");
  device_info_uint(device, CL_DEVICE_VENDOR_ID, "CL_DEVICE_VENDOR_ID");
  device_info_string(device, CL_DEVICE_VERSION, "CL_DEVICE_VERSION");
  device_info_string(device, CL_DRIVER_VERSION, "CL_DRIVER_VERSION");
  device_info_uint(device, CL_DEVICE_ADDRESS_BITS, "CL_DEVICE_ADDRESS_BITS");
  device_info_bool(device, CL_DEVICE_AVAILABLE, "CL_DEVICE_AVAILABLE");
  device_info_bool(device, CL_DEVICE_ENDIAN_LITTLE, "CL_DEVICE_ENDIAN_LITTLE");
  device_info_ulong(device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, "CL_DEVICE_GLOBAL_MEM_CACHE_SIZE");
  device_info_ulong(device, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, "CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE");
  device_info_ulong(device, CL_DEVICE_GLOBAL_MEM_SIZE, "CL_DEVICE_GLOBAL_MEM_SIZE");
  device_info_bool(device, CL_DEVICE_IMAGE_SUPPORT, "CL_DEVICE_IMAGE_SUPPORT");
  device_info_ulong(device, CL_DEVICE_LOCAL_MEM_SIZE, "CL_DEVICE_LOCAL_MEM_SIZE");
  device_info_ulong(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, "CL_DEVICE_MAX_CLOCK_FREQUENCY");
  device_info_ulong(device, CL_DEVICE_MAX_COMPUTE_UNITS, "CL_DEVICE_MAX_COMPUTE_UNITS");
  device_info_ulong(device, CL_DEVICE_MAX_CONSTANT_ARGS, "CL_DEVICE_MAX_CONSTANT_ARGS");
  device_info_ulong(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE");
  device_info_uint(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS");
  device_info_uint(device, CL_DEVICE_MEM_BASE_ADDR_ALIGN, "CL_DEVICE_MEM_BASE_ADDR_ALIGN");
  device_info_uint(device, CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE, "CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE");
  device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR");
  device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT");
  device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT");
  device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG");
  device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT");
  device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE");

  {
    cl_command_queue_properties ccp;
    clGetDeviceInfo(device, CL_DEVICE_QUEUE_PROPERTIES, sizeof(cl_command_queue_properties), &ccp, NULL);
    printf("%-40s = %s\n", "Command queue out of order? ", ((ccp & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)?"true":"false"));
    printf("%-40s = %s\n", "Command queue profiling enabled? ", ((ccp & CL_QUEUE_PROFILING_ENABLE)?"true":"false"));
  }
}

#endif
