# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 10:29:43 2020

@author: lenovo
"""


import logging

 

def sample_function(secret_parameter):

    logger = logging.getLogger(__name__)  # __name__=projectA.moduleB

    logger.debug("Going to perform magic with '%s'",  secret_parameter)

    ...

    try:

        result = do_magic(secret_parameter)

    except IndexError:

        logger.exception("OMG it happened again, someone please tell Laszlo")

    except:

        logger.info("Unexpected exception", exc_info=True)

        raise

    else:

        logger.info("Magic with '%s' resulted in '%s'", secret_parameter, result, stack_info=True)