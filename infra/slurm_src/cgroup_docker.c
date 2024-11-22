
/*
 * This file is a cgroup plugin for Slurm that can be used inside Docker containers.
 *
 * The plugin is based on the cgroup plugin provided by Slurm, and it is modified to work
 * inside Docker containers without privileged mode.
 *
 * The plugin is used to create cgroups for Slurm jobs and tasks, and to set resource limits
 * for them. The plugin is used to track resource usage of the tasks and jobs.
 * The plugin is used to track the OOM events and to kill the tasks that consume too much memory.
 *
 * Given that the plugin is used inside Docker containers, it is not possible to use the cgroup
 * filesystem to create cgroups. So most of the functionality is currently a no-op.
 */

#ifndef CGROUP_DOCKER_C
#define CGROUP_DOCKER_C

#include <slurm/slurm.h>
#include <slurm/slurm_errno.h>

// Includes for pid_t
#include <sys/types.h>

#define USEC_IN_SEC 1000000

/* Current supported cgroup controller types */
typedef enum {
	CG_TRACK,
	CG_CPUS,
	CG_MEMORY,
	CG_DEVICES,
	CG_CPUACCT,
	CG_CTL_CNT
} cgroup_ctl_type_t;

typedef enum {
	CG_LEVEL_ROOT,
	CG_LEVEL_SLURM,
	CG_LEVEL_USER,
	CG_LEVEL_JOB,
	CG_LEVEL_STEP,
	CG_LEVEL_STEP_SLURM,
	CG_LEVEL_STEP_USER,
	CG_LEVEL_TASK,
	CG_LEVEL_SYSTEM,
	CG_LEVEL_CNT
} cgroup_level_t;

typedef void stepd_step_rec_t;

typedef enum {
	DEV_TYPE_NONE,
	DEV_TYPE_BLOCK,
	DEV_TYPE_CHAR,
} gres_device_type_t;

typedef struct {
	uint32_t major;
	uint32_t minor;
	gres_device_type_t type;
} gres_device_id_t;

typedef struct {
	/* extra info */
	stepd_step_rec_t *step;
	uint32_t taskid;
	/* task cpuset */
	char *allow_cores;
	char *allow_mems;
	size_t cores_size;
	size_t mems_size;
	/* task devices */
	bool allow_device;
	gres_device_id_t device;
	/* jobacct memory */
	uint64_t limit_in_bytes;
	uint64_t soft_limit_in_bytes;
	uint64_t memsw_limit_in_bytes;
	uint64_t swappiness;
} cgroup_limits_t;

typedef struct {
	uint64_t step_mem_failcnt;
	uint64_t step_memsw_failcnt;
	uint64_t job_mem_failcnt;
	uint64_t job_memsw_failcnt;
	uint64_t oom_kill_cnt;
} cgroup_oom_t;

typedef struct {
	uint64_t usec;
	uint64_t ssec;
	uint64_t total_rss;
	uint64_t total_pgmajfault;
	uint64_t total_vmem;
} cgroup_acct_t;

typedef enum {
	CG_MEMCG_SWAP
} cgroup_ctl_feature_t;

const char plugin_name[] = "Cgroup Docker plugin";
const char plugin_type[] = "cgroup/docker";
const uint32_t plugin_version = SLURM_VERSION_NUMBER;

extern int init(void)
{
    return SLURM_SUCCESS;
}

extern int fini(void)
{
    return SLURM_SUCCESS;
}

extern int cgroup_p_initialize(cgroup_ctl_type_t ctl)
{
    return SLURM_SUCCESS;
}

extern int cgroup_p_system_create(cgroup_ctl_type_t ctl)
{
	return SLURM_SUCCESS;
}

extern int cgroup_p_system_addto(cgroup_ctl_type_t ctl, pid_t *pids, int npids)
{
	return SLURM_SUCCESS;
}

extern int cgroup_p_system_destroy(cgroup_ctl_type_t ctl)
{
	return SLURM_SUCCESS;
}

extern int cgroup_p_step_create(cgroup_ctl_type_t ctl, stepd_step_rec_t *step)
{
    return SLURM_SUCCESS;
}

extern int cgroup_p_step_addto(cgroup_ctl_type_t ctl, pid_t *pids, int npids)
{
    return SLURM_SUCCESS;
}

extern int cgroup_p_step_get_pids(pid_t **pids, int *npids)
{
	return SLURM_SUCCESS;
}

extern int cgroup_p_step_suspend(void)
{
    return SLURM_SUCCESS;
}

extern int cgroup_p_step_resume(void)
{
    return SLURM_SUCCESS;
}

extern int cgroup_p_step_destroy(cgroup_ctl_type_t ctl)
{
    return SLURM_SUCCESS;
}

extern bool cgroup_p_has_pid(pid_t pid)
{
    return false;
}

extern int cgroup_p_constrain_set(cgroup_ctl_type_t ctl, cgroup_level_t level,
				  cgroup_limits_t *limits)
{
    return SLURM_SUCCESS;
}

extern int cgroup_p_constrain_apply(cgroup_ctl_type_t ctl, cgroup_level_t level,
                                    uint32_t task_id)
{
    return SLURM_SUCCESS;
}

extern cgroup_limits_t *cgroup_p_constrain_get(cgroup_ctl_type_t ctl,
					       cgroup_level_t level)
{
    return NULL;
}

extern int cgroup_p_step_start_oom_mgr(void)
{
	/* Just return, no need to start anything. */
	return SLURM_SUCCESS;
}

extern cgroup_oom_t *cgroup_p_step_stop_oom_mgr(stepd_step_rec_t *step)
{
    return NULL;
}

extern int cgroup_p_task_addto(cgroup_ctl_type_t ctl, stepd_step_rec_t *step,
			       pid_t pid, uint32_t task_id)
{
    return SLURM_SUCCESS;
}


extern cgroup_acct_t *cgroup_p_task_get_acct_data(uint32_t task_id)
{
    return NULL;
}

extern long int cgroup_p_get_acct_units(void)
{
	/* usec and ssec from cpuacct.stat are provided in micro-seconds. */
	return (long int)USEC_IN_SEC;
}

extern bool cgroup_p_has_feature(cgroup_ctl_feature_t f)
{
    return false;
}

#endif
