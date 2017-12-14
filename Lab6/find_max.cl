/*
 * Placeholder OpenCL kernel
 * https://stackoverflow.com/questions/36465581/opencl-find-max-in-array
 */

__kernel void find_max(__global unsigned int *data, const unsigned int length)
{ 
  /*unsigned int pos = 0;
  unsigned int val;

  //Something should happen here

  data[get_global_id(0)]=get_global_id(0);
  */
  
  int id = get_local_id(0);
  
  for(int dataSize=length; dataSize>0; dataSize/=2){
	  if(id<dataSize){
		int a = data[id];
		int b = data[id+dataSize/2];
		data[id]=a>b?a:b;
	  }
  }
  
}
