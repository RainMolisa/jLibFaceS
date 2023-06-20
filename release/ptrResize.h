#pragma once
template<typename A,typename B>
void ptrResize(A* InIg,int InRows,int InCols,
	B* OtIg,int OtRows,int OtCols);
template<typename A,typename B>
void ptrResize160(A* InIg,int InRows,int InCols,B* OtIg);

/*
template<typename A,typename B>
void ptrResize(A* InIg,int InRows,int InCols,
	B* OtIg,int OtRows,int OtCols){
	for(int oy=0;oy<OtRows;oy++){
		float iy=float(InRows)*(float(oy)/float(OtRows));
		float y0=int(iy);
		float y=iy-y0;
		if(y<=0.0){
			for(int ox=0;ox<OtCols;ox++){
				float ix=float(InCols)*(float(ox)/float(OtCols));
				float x0=int(ix);
				float x=ix-x0;
				//printf("%f x0=%f x=%f\n", ix, x0, x);
				if(x<=0.0){
					OtIg[oy*OtCols+ox]=InIg[int(y0 *InCols+ x0)];
				}else{
					float v = (1.0 - x) * float(InIg[int(y0 * InCols + x0)]) + x * float(InIg[int(y0 * InCols + x0 + 1)]);
					OtIg[oy * OtCols + ox] = B(v+0.5);
				}
			}
		}else{
			for(int ox=0;ox<OtCols;ox++){
				float ix=float(InCols)*(float(ox)/float(OtCols));
				float x0=int(ix);
				float x=ix-x0;
				if(x<=0.0){
					float v= (1.0 - y) * float(InIg[int((y0)*InCols + x0)]) + y * float(InIg[int((y0 + 1) * InCols + x0)]);
					OtIg[oy * OtCols + ox] = B(v + 0.5);
				}else{
					float v=(1.0-x)*(1.0-y)* float(InIg[int((y0 )*InCols+ x0 )])+
						(1.0-x)*y* float(InIg[int((y0+1) *InCols+ x0 )])+
						x*(1.0-y)*float(InIg[int((y0 )*InCols+ x0+1)])+
						x*y*float(InIg[int((y0+1) *InCols+ x0+1)]);
					OtIg[oy*OtCols+ox]= B(v + 0.5);
				}
			}
		}
	}
}
*/
template<typename A,typename B>
void ptrResize(A* InIg,int InRows,int InCols,
	B* OtIg,int OtRows,int OtCols){
	typedef float biArr[4];
	biArr *rInx=new biArr[OtRows];
	biArr *cInx=new biArr[OtCols];
	for(int i=0;i<OtCols;i++){
		float val=float(i*InCols)/float(OtCols);
		float intpt=int(val);
		float decpt=val-intpt;
		if(decpt<=0.0){
			cInx[i][0]=(1.0-decpt);
			cInx[i][1]=intpt;
			cInx[i][2]=decpt;
			cInx[i][3]=intpt;
		}else{
			cInx[i][0]=(1.0-decpt);
			cInx[i][1]=intpt;
			cInx[i][2]=decpt;
			cInx[i][3]=intpt+1;
		}
	}
	for(int i=0;i<OtRows;i++){
		float val=float(i*InRows)/float(OtRows);
		float intpt=int(val);
		float decpt=val-intpt;
		if(decpt<=0.0){
			rInx[i][0]=(1.0-decpt);
			rInx[i][1]=intpt;
			rInx[i][2]=decpt;
			rInx[i][3]=intpt;
		}else{
			rInx[i][0]=(1.0-decpt);
			rInx[i][1]=intpt;
			rInx[i][2]=decpt;
			rInx[i][3]=intpt+1;
		}
	}
	for(int oy=0;oy<OtRows;oy++){
		for(int ox=0;ox<OtCols;ox++){
			float v=cInx[ox][0]*rInx[oy][0]*float(InIg[int(rInx[oy][1]*InCols+cInx[ox][1])])+
				cInx[ox][0]*rInx[oy][2]*float(InIg[int(rInx[oy][3]*InCols+cInx[ox][1])])+
				cInx[ox][2]*rInx[oy][0]*float(InIg[int(rInx[oy][1]*InCols+cInx[ox][3])])+
				cInx[ox][2]*rInx[oy][2]*float(InIg[int(rInx[oy][3]*InCols+cInx[ox][3])]);
			OtIg[oy*OtCols+ox]= B(v + 0.5);
		}
	}
	delete[] rInx;
	delete[] cInx;
}
template<typename A,typename B>
void ptrResize2(A* InIg,int InRows,int InCols,
	B* OtIg,int OtRows,int OtCols){
	//printf("a\n");
	int *rInx=new int[OtRows];
	int *cInx=new int[OtCols];
	for(int i=0;i<OtCols;i++){
		float val=float(i*InCols)/float(OtCols)+0.5;
		cInx[i]=val;
	}
	for(int i=0;i<OtRows;i++){
		float val=float(i*InRows)/float(OtRows)+0.5;
		rInx[i]=val;
	}
	for(int oy=0;oy<OtRows;oy++){
		int i=oy*OtCols;
		for(int ox=0;ox<OtCols;ox++){
			OtIg[i+ox]=InIg[rInx[oy]*InCols+cInx[ox]];
		}
	}
	delete[] rInx;
	delete[] cInx;
}

template<typename A,typename B>
void ptrResize160(A* InIg,int InRows,int InCols,B* OtIg){
	ptrResize<A,B>(InIg,InRows,InCols,OtIg,160,160);
}