/*
 * stack.c
 *
 *  Created on: 18 Oct 2011
 *  Copyright 2011 Nicolas Melot
 *
 * This file is part of TDDD56.
 * 
 *     TDDD56 is free software: you can redistribute it and/or modify
 *     it under the terms of the GNU General Public License as published by
 *     the Free Software Foundation, either version 3 of the License, or
 *     (at your option) any later version.
 * 
 *     TDDD56 is distributed in the hope that it will be useful,
 *     but WITHOUT ANY WARRANTY without even the implied warranty of
 *     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *     GNU General Public License for more details.
 * 
 *     You should have received a copy of the GNU General Public License
 *     along with TDDD56. If not, see <http://www.gnu.org/licenses/>.
 * 
 */

#ifndef DEBUG
#define NDEBUG
#endif

#include <assert.h>
#include <pthread.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include "stack.h"
#include "non_blocking.h"

#if NON_BLOCKING == 0
#warning Stacks are synchronized through locks
#else
#if NON_BLOCKING == 1
#warning Stacks are synchronized through hardware CAS
#else
#warning Stacks are synchronized through lock-based CAS
#endif
#endif



void
print_stack(stack_t *stack)
{
	node_t *e=stack->head;
	while (e != NULL){
		printf("v->%i\n",e->value);
		e=e->next;
	}
}

int
stack_check(stack_t *stack)
{
// Do not perform any sanity check if performance is bein measured
#if MEASURE == 0
	// Use assert() to check if your stack is in a state that makes sens
	// This test should always pass 
	assert(1 == 1);

	// This test fails if the task is not allocated or if the allocation failed
	assert(stack != NULL);
#endif
	// The stack is always fine
	return 1;
}

int /* Return the type you prefer */
stack_push(stack_t *stack, int value/* Make your own signature */)
{
#if NON_BLOCKING == 0
  // Implement a lock_based stack
  /** add by us**/
	node_t* new_node = (node_t *)malloc(sizeof(node_t));
	new_node->value = value;
	pthread_mutex_lock(&stack->lock);
	if(!stack->head){
		new_node->next = NULL;	
	}else{
		new_node->next = stack->head;
	}
	stack->head = new_node;
	pthread_mutex_unlock(&stack->lock);
		
  /** **/
#elif NON_BLOCKING == 1
  // Implement a harware CAS-based stack
  int condition=1;
  node_t* new_node = (node_t *)malloc(sizeof(node_t));
  new_node->value = value;
  
  while(condition){
	node_t *old_top = stack->head;
	new_node->next=old_top;
	if(cas(&stack->head,old_top,new_node)==old_top){
		condition=0;
	}
	
  }
  
#else
  /*** Optional ***/
  // Implement a software CAS-based stack
#endif

  // Debug practice: you can check if this operation results in a stack in a consistent check
  // It doesn't harm performance as sanity check are disabled at measurement time
  // This is to be updated as your implementation progresses
  stack_check((stack_t*)1);

  return value;
}

int /* Return the type you prefer */
stack_pop(stack_t *stack/* Make your own signature */)
{
#if NON_BLOCKING == 0
  // Implement a lock_based stack
    /** add by us**/
    int result;
	pthread_mutex_lock(&stack->lock);
	
	if(stack->head == NULL){
		result=-1;
	}else{
		node_t* old_node = stack->head;
		stack->head = old_node->next;
		result = old_node->value;
		free(old_node);
	}
	pthread_mutex_unlock(&stack->lock);
		
	/** **/
#elif NON_BLOCKING == 1
  // Implement a harware CAS-based stack
	/** add by us **/
	int condition = 1;
	while( condition ) {
		node_t *head = stack->head;
		if (head == NULL ) {
			return -1;
		}
		node_t* new_head = head->next;
		if (cas(&stack->head,head,new_head)==head) {
			condition = 0;
		}
	}
	/** **/
#else
  /*** Optional ***/
  // Implement a software CAS-based stack
#endif

  return result;
}

