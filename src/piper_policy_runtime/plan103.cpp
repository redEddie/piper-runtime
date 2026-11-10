// CPU port of gentle-placement/control_lab/smooth_plan.py (double precision).
// No device I/O. Fixed six-joint chain; 100ms motion plus checked stop backup.
#include <algorithm>
#include <cmath>
#include <cstring>

namespace {
using V = double[6];
double clip(double x, double lo, double hi) { return std::min(std::max(x, lo), hi); }
void mv(const double* r, const double* x, double* y) {
    for (int i=0;i<3;i++) y[i]=r[3*i]*x[0]+r[3*i+1]*x[1]+r[3*i+2]*x[2];
}
void mm(const double* a,const double* b,double* c) {
    double t[9]={};
    for(int i=0;i<3;i++)for(int j=0;j<3;j++)for(int k=0;k<3;k++)t[3*i+j]+=a[3*i+k]*b[3*k+j];
    std::memcpy(c,t,sizeof(t));
}
// kin: six [offset(3), body rotation(9), pivot(3), axis(3), reference(1)], TCP(3).
double floor_fk(const double* kin,const double* q,const double* n,double* jac) {
    double r[9]={1,0,0,0,1,0,0,0,1},p[3]={},origins[6][3],axes[6][3];
    for(int i=0;i<6;i++) {
        const double* k=kin+19*i; double temp[3]; mv(r,k,temp);
        for(int j=0;j<3;j++)p[j]+=temp[j];
        mm(r,k+3,r); mv(r,k+12,temp);
        for(int j=0;j<3;j++)origins[i][j]=p[j]+temp[j];
        mv(r,k+15,axes[i]);
        double x=k[15],y=k[16],z=k[17],c=std::cos(q[i]-k[18]),s=std::sin(q[i]-k[18]),d=1-c;
        double rot[9]={c+x*x*d,x*y*d-z*s,x*z*d+y*s,
                       y*x*d+z*s,c+y*y*d,y*z*d-x*s,
                       z*x*d-y*s,z*y*d+x*s,c+z*z*d};
        mm(r,rot,r); mv(r,k+12,temp);
        for(int j=0;j<3;j++)p[j]=origins[i][j]-temp[j];
    }
    double temp[3];mv(r,kin+114,temp);for(int j=0;j<3;j++)p[j]+=temp[j];
    if(jac)for(int i=0;i<6;i++) {
        double a=p[0]-origins[i][0],b=p[1]-origins[i][1],c=p[2]-origins[i][2];
        jac[i]=(axes[i][1]*c-axes[i][2]*b)*n[0]+(axes[i][2]*a-axes[i][0]*c)*n[1]+(axes[i][0]*b-axes[i][1]*a)*n[2];
    }
    return p[0]*n[0]+p[1]*n[1]+p[2]*n[2];
}
// settings: dt,speed,acceleration,jerk,lag,extra,margin,horizon,weight,smooth,
// floor_offset,normal(3),tolerance,exit_margin.
void terminal(double v,double target,const double* s,double& lo,double& hi) {
    lo=hi=target;double h=s[3]*s[0]*s[0];int count=std::ceil(s[2]/(s[3]*s[0]));
    for(int m=1;m<count;m++) {
        double center=(target+m*v)/(m+1);
        lo=std::max(lo,center-h*m/2);hi=std::min(hi,center+h*m/2);
    }
}
void speed_bounds(double v,double a,const double* s,double& lo,double& hi) {
    double dt=s[0],h=s[3]*dt*dt,tmp,left,right;
    lo=std::max(-s[1],std::max(v-s[2]*dt,v+dt*a-h));
    hi=std::min(s[1],std::min(v+s[2]*dt,v+dt*a+h));
    terminal(v,-s[1],s,left,tmp); terminal(v,s[1],s,tmp,right);
    lo=std::max(lo,left); hi=std::min(hi,right);
}
double velocity_step(double v,double a,double target,const double* s) {
    double lo,hi,left,right; speed_bounds(v,a,s,lo,hi);terminal(v,target,s,left,right);
    return clip(target>=v?right:left,lo,hi);
}
void position_bounds(double ref,double v,double lower,double upper,const double* s,double& lo,double& hi) {
    double dt=s[0],duration=s[1]/s[2]+3*s[2]/s[3]+4*dt,h=s[3]*dt*dt;
    lo=(lower-ref)/(dt+duration);hi=(upper-ref)/(dt+duration);
    for(int m=1;m<std::ceil(s[2]/(s[3]*dt));m++) {
        double denom=dt+duration*(m+1);
        lo=std::max(lo,((lower-ref)+duration*m*v-duration*h*m*(m+1)/2)/denom);
        hi=std::min(hi,((upper-ref)+duration*m*v+duration*h*m*(m+1)/2)/denom);
    }
}
double projected(double z,double lo,double hi,double slo,double shi,double w) {
    return clip(z<slo?(z+w*slo)/(1+w):(z>shi?(z+w*shi)/(1+w):z),lo,hi);
}
bool project(const double* target,const double* lo,const double* hi,const double* slo,const double* shi,
             double w,const double* normal,double minimum,double* out) {
    double maximum=0,base=0;
    for(int j=0;j<6;j++) {
        if(lo[j]>hi[j])return false;
        maximum+=normal[j]*(normal[j]>=0?hi[j]:lo[j]);
        out[j]=projected(target[j],lo[j],hi[j],slo[j],shi[j],w);base+=normal[j]*out[j];
    }
    if(maximum<minimum-1e-6)return false;
    if(base>=minimum)return true;
    auto dot=[&](double a) {double d=0;for(int j=0;j<6;j++)d+=normal[j]*projected(target[j]+a*normal[j],lo[j],hi[j],slo[j],shi[j],w);return d;};
    double left=0,right=1;
    for(int k=0;k<40;k++)if(dot(right)<minimum)right*=2;
    for(int k=0;k<32;k++){double m=(left+right)/2;if(dot(m)<minimum)left=m;else right=m;}
    for(int j=0;j<6;j++)out[j]=projected(target[j]+right*normal[j],lo[j],hi[j],slo[j],shi[j],w);
    return true;
}
}

static int plan_impl(const double* kin,const double* q,const double* qdot,const double* initial,
                       const double* target,const double* old_target,int hold,const double* lower,
                       const double* upper,const double* s,int steps,int stop_steps,double* plan,double* detail,bool sampled) {
    if(steps<1||stop_steps<1||steps>100||stop_steps>1000)return -1;
    double st[18];std::memcpy(st,initial,sizeof(st));bool ok=true;double dt=s[0];
    for(int i=0;i<steps+stop_steps;i++) {
        V vn;
        if(i>=steps) {
            for(int j=0;j<6;j++)vn[j]=velocity_step(st[6+j],st[12+j],0,s);
        } else {
            V desired,normal,preferred,lo,hi,slo,shi; double z=floor_fk(kin,st,s+11,normal),dot=0,norm=0;
            for(int j=0;j<6;j++) {
                desired[j]=sampled?target[6*i+j]:clip(i<hold?old_target[j]:target[j],(lower[j]-st[j])/.5,(upper[j]-st[j])/.5);
                dot+=normal[j]*desired[j];norm+=normal[j]*normal[j];
            }
            double approach=-std::max(z-s[10]-.001,0.)/.5;
            double correction=sampled?0.:std::max(approach-dot,0.)/std::max(norm,1e-8);
            for(int j=0;j<6;j++) {
                desired[j]+=normal[j]*correction;
                double shaped=sampled?desired[j]:velocity_step(st[6+j],st[12+j],desired[j],s);
                double measured=q[j]+std::min(i*dt,.05)*qdot[j],error=st[j]-measured,band=s[4]-s[6];
                double recovery=qdot[j]-(error-clip(error,-band,band))/s[7];
                preferred[j]=((std::abs(error)>s[4]?recovery:shaped)+s[9]*st[6+j])/(1+s[9]);
                double adt=s[2]*dt;
                lo[j]=std::max(std::max(-s[1],(lower[j]-st[j])/dt),st[6+j]-adt);
                hi[j]=std::min(std::min(s[1],(upper[j]-st[j])/dt),st[6+j]+adt);
                lo[j]=std::max(lo[j],adt-std::sqrt(adt*adt+2*s[2]*std::max(0.,st[j]-lower[j])));
                hi[j]=std::min(hi[j],-adt+std::sqrt(adt*adt+2*s[2]*std::max(0.,upper[j]-st[j])));
                double center=st[6+j]+dt*st[12+j],radius=s[3]*dt*dt;
                lo[j]=std::max(lo[j],center-radius);hi[j]=std::min(hi[j],center+radius);
                double qnext=measured+dt*qdot[j],cap=s[4]+s[5];
                lo[j]=std::max(lo[j],(std::max(measured,qnext)-cap-st[j])/dt);
                hi[j]=std::min(hi[j],(std::min(measured,qnext)+cap-st[j])/dt);
                slo[j]=(measured+s[7]*qdot[j]-band-st[j])/s[7];
                shi[j]=(measured+s[7]*qdot[j]+band-st[j])/s[7];
                double sl,sh,pl,ph;speed_bounds(st[6+j],st[12+j],s,sl,sh);
                lo[j]=std::max(lo[j],sl);hi[j]=std::min(hi[j],sh);
                if(!sampled) {
                    position_bounds(st[j],st[6+j],lower[j],upper[j],s,pl,ph);
                    lo[j]=std::max(lo[j],pl);hi[j]=std::min(hi[j],ph);
                }
            }
            double distance=z-s[10],minimum=std::max(-distance/dt,-5*(distance-.001));
            bool feasible=project(preferred,lo,hi,slo,shi,s[8],normal,minimum,vn);
            ok=ok&&feasible;
            if(!feasible)for(int j=0;j<6;j++)vn[j]=velocity_step(st[6+j],st[12+j],0,s);
        }
        for(int j=0;j<6;j++) {st[12+j]=(vn[j]-st[6+j])/dt;st[6+j]=vn[j];st[j]+=dt*vn[j];}
        std::memcpy(plan+18*i,st,sizeof(st));
    }
    bool within=true,finite=true;double minfloor=INFINITY,vmax=0,amax=0;
    for(int i=0;i<steps+stop_steps;i++) {
        const double* row=plan+18*i;
        for(int j=0;j<18;j++)finite=finite&&std::isfinite(row[j]);
        for(int j=0;j<6;j++)within=within&&(row[j]>=lower[j]-s[14])&&(row[j]<=upper[j]+s[14]);
        minfloor=std::min(minfloor,floor_fk(kin,row,s+11,nullptr)-s[10]);
    }
    for(int j=0;j<6;j++){vmax=std::max(vmax,std::abs(st[6+j]));amax=std::max(amax,std::abs(st[12+j]));}
    detail[0]=ok;detail[1]=within;detail[2]=minfloor;detail[3]=vmax;detail[4]=amax;
    return ok&&within&&minfloor>=-s[14]&&vmax<1e-5&&amax<1e-3&&finite;
}

extern "C" int plan103(const double* kin,const double* q,const double* qdot,const double* initial,
                       const double* target,const double* old_target,int hold,const double* lower,
                       const double* upper,const double* s,int steps,int stop_steps,double* plan,double* detail) {
    return plan_impl(kin,q,qdot,initial,target,old_target,hold,lower,upper,s,steps,stop_steps,plan,detail,false);
}

// Posture mode: time-optimal nominal velocities, followed by the same QP hard
// limits and full nonlinear stop certificate. The conservative training-only
// mechanical taper is replaced by this checked stop, never by relaxed bounds.
extern "C" int plan103_samples(const double* kin,const double* q,const double* qdot,const double* initial,
                       const double* samples,const double* old_target,int hold,const double* lower,
                       const double* upper,const double* s,int steps,int stop_steps,double* plan,double* detail) {
    return plan_impl(kin,q,qdot,initial,samples,old_target,hold,lower,upper,s,steps,stop_steps,plan,detail,true);
}
