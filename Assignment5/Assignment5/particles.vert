#extension GL_EXT_gpu_shader4 : enable

uniform samplerBuffer AABBmin;
uniform samplerBuffer AABBmax;

uniform int aabbminmax;

vec4 colorCode(float value)
{
	value *= 0.2;

	vec4 retVal;
	retVal.x = 2.0 * value * mix(0.0, 1.0, max(0.0, 1.0 - value)); 
	retVal.y = 2.0 * value * min( mix(0.0, 1.0, max(0.0, value) / 0.5), mix(0.0, 1.0, max(0.0, (1.0 - value) / 0.5))); 
	retVal.z = 1.f;
	retVal.w = 0.4;
	
	return retVal;
}

varying vec4 col;

void main() {

	vec3 c000 = texelFetchBuffer(AABBmin, gl_InstanceID).xyz;
	vec3 c111 = texelFetchBuffer(AABBmax, gl_InstanceID).xyz;

  vec3 size = c111 - c000;

  vec3 c100 = c000 + vec3(size.x, 0, 0);
  vec3 c010 = c000 + vec3(0, size.y, 0);
  vec3 c001 = c000 + vec3(0, 0, size.z);

  vec3 c110 = c000 + vec3(size.x, size.y, 0);
  vec3 c011 = c000 + vec3(0, size.y, size.z);
  vec3 c101 = c000 + vec3(size.x, 0, size.z);


  int vid = gl_VertexID % 24;
  vec3 v;

  if (vid == 0) v = c000;
  else if (vid == 1) v = c100;

  else if (vid == 2) v = c000;
  else if (vid == 3) v = c010;

  else if (vid == 4) v = c000;
  else if (vid == 5) v = c001;

  else if (vid == 6) v = c111;
  else if (vid == 7) v = c011;

  else if (vid == 8) v = c111;
  else if (vid == 9) v = c101;

  else if (vid == 10) v = c111;
  else if (vid == 11) v = c110;

  else if (vid == 12) v = c010;
  else if (vid == 13) v = c011;

  else if (vid == 14) v = c011;
  else if (vid == 15) v = c001;

  else if (vid == 16) v = c001;
  else if (vid == 17) v = c101;

  else if (vid == 18) v = c101;
  else if (vid == 19) v = c100;

  else if (vid == 20) v = c100;
  else if (vid == 21) v = c110;

  else if (vid == 22) v = c110;
  else if (vid == 23) v = c010;
    

  col = vec4(0,1,0,1);
  gl_Position = gl_ModelViewProjectionMatrix * vec4(v, 1);
}
