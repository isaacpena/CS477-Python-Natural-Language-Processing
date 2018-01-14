from nltk.compat import python_2_unicode_compatible

printed = False

@python_2_unicode_compatible
class FeatureExtractor(object):
    @staticmethod
    def _check_informative(feat, underscore_is_informative=False):
        """
        Check whether a feature is informative
        """

        if feat is None:
            return False

        if feat == '':
            return False

        if not underscore_is_informative and feat == '_':
            return False

        return True
   
    @staticmethod 
    def find_left_right_valencies(idx, arcs):
        leftval = 0
        rightval = 0
        for (wi, r, wj) in arcs:
            if wi == idx:
                if (wj > wi): 
                    rightval += 1
                if (wj < wi):
                    leftval += 1
        return leftval, rightval

    @staticmethod
    def find_left_right_dependencies(idx, arcs):
        left_most = 1000000
        right_most = -1
        dep_left_most = ''
        dep_right_most = ''
        for (wi, r, wj) in arcs:
            if wi == idx:
                if (wj > wi) and (wj > right_most):
                    right_most = wj
                    dep_right_most = r
                if (wj < wi) and (wj < left_most):
                    left_most = wj
                    dep_left_most = r
        return dep_left_most, dep_right_most, left_most, right_most

    @staticmethod
    def extract_features(tokens, buffer, stack, arcs):
        """
        This function returns a list of string features for the classifier

        :param tokens: nodes in the dependency graph
        :param stack: partially processed words
        :param buffer: remaining input words
        :param arcs: partially built dependency tree

        :return: list(str)
        """

        """
        Think of some of your own features here! Some standard features are
        described in Table 3.2 on page 31 of Dependency Parsing by Kubler,
        McDonald, and Nivre

        [http://books.google.com/books/about/Dependency_Parsing.html?id=k3iiup7HB9UC]
        """

        result = []


        global printed
        if not printed:
            print("This is not a very good feature extractor!")
            printed = True

        # My feats:
        # POS to STK[0]: increases UAS from .23 to .44, LAS from .125 to .31
        # POS to BUF[0]: increases UAS from .23 to .53, LAS from .125 to .37
        # POS to both STK[0] and BUF[0]: increases UAS from .23 to .70, LAS from .125 to .59 

        # an example set of features:
        stk0word = 0
        buf0word = 0
        buf1word = 0
        
        stk0pos = 0
        stk1pos = 0    
        buf0pos = 0
        buf1pos = 0
        
        stkldep = ''
        stkrdep = ''
        bufldep = ''
        bufrdep = ''

        stk_idx0 = 0
        stk_idx1 = 0
        buf_idx0 = 0
        buf_idx1 = 0
     
        stk0_token = None
        stk1_token = None
        buf0_token = None
        buf1_token = None

        if stack:
            stk_idx0 = stack[-1]
            stk0_token = tokens[stk_idx0]
        
            if FeatureExtractor._check_informative(stk0_token['word'], True):
                result.append('STK_0_FORM_' + stk0_token['word'])
                stk0word = 1
            if FeatureExtractor._check_informative(stk0_token['tag'], True):
                result.append('STK_0_POS_' + stk0_token['tag'])
                stk0pos = 1
            if FeatureExtractor._check_informative(stk0_token['lemma'], True):
                result.append('STK_0_LEMMA_' + stk0_token['lemma'])
            if FeatureExtractor._check_informative(stk0_token['ctag'], True):
                result.append('STK_0_CPOS_' + stk0_token['ctag'])
                
            if 'feats' in stk0_token and FeatureExtractor._check_informative(stk0_token['feats']):
                feats = stk0_token['feats'].split("|")
                for feat in feats:
                    result.append('STK_0_FEATS_' + feat)

            # Left most, right most dependency of stack[0]
            dep_left_most, dep_right_most, stk0_ldep_ind, stk0_rdep_ind = FeatureExtractor.find_left_right_dependencies(stk_idx0, arcs)

            if FeatureExtractor._check_informative(dep_left_most):
                result.append('STK_0_LDEP_' + dep_left_most)
                stkldep = dep_left_most
                result.append('STK_0_LDEP_POS_' + tokens[stk0_ldep_ind]['tag'])
            if FeatureExtractor._check_informative(dep_right_most):
                result.append('STK_0_RDEP_' + dep_right_most)
                stkrdep = dep_right_most
                result.append('STK_0_RDEP_POS_' + tokens[stk0_rdep_ind]['tag'])

            lval, rval = FeatureExtractor.find_left_right_valencies(stk_idx0, arcs)
            if lval != 0:
                result.append('STK_0_LVAL_POS_' + stk0_token['tag'] + '_' + str(lval))

        if buffer:
            buf_idx0 = buffer[0]
            buf0_token = tokens[buf_idx0]

            if FeatureExtractor._check_informative(buf0_token['word'], True):
                result.append('BUF_0_FORM_' + buf0_token['word'])
                buf0word = 1
            if FeatureExtractor._check_informative(buf0_token['tag'], True):
                result.append('BUF_0_POS_' + buf0_token['tag'])
                buf0pos = 1
            if FeatureExtractor._check_informative(buf0_token['lemma'], True):
                result.append('BUF_0_LEMMA_' + buf0_token['lemma'])
            if FeatureExtractor._check_informative(buf0_token['ctag'], True):
                result.append('BUF_0_CPOS_' + buf0_token['ctag'])


            if 'feats' in buf0_token and FeatureExtractor._check_informative(buf0_token['feats']):
                feats = buf0_token['feats'].split("|")
                for feat in feats:
                    result.append('BUF_0_FEATS_' + feat)
    
            # Left most, right most dependency of buf[0]
            dep_left_most, dep_right_most, buf0_ldep_ind, buf0_rdep_ind = FeatureExtractor.find_left_right_dependencies(buf_idx0, arcs)

            if FeatureExtractor._check_informative(dep_left_most):
                result.append('BUF_0_LDEP_' + dep_left_most)
                bufldep = dep_left_most
                result.append('BUF_0_LDEP_POS_' + tokens[buf0_ldep_ind]['tag'])
                    
            if FeatureExtractor._check_informative(dep_right_most):
                result.append('BUF_0_RDEP_' + dep_right_most)
                bufrdep = dep_right_most


            lval, rval = FeatureExtractor.find_left_right_valencies(buf_idx0, arcs)
            if lval != 0:
                result.append('BUF_0_LVAL_POS_' + buf0_token['tag'] + '_' + str(lval))

        if len(stack) > 1:
            stk_idx1 = stack[-2]
            stk1_token = tokens[stk_idx1]
 
            if FeatureExtractor._check_informative(stk1_token['tag'], True):
                result.append('STK_1_POS_' + stk1_token['tag'])
                stk1pos = 1

        if len(buffer) > 1:
            buf_idx1 = buffer[1]
            buf1_token = tokens[buf_idx1]
            
            if FeatureExtractor._check_informative(buf1_token['word'], True):
                result.append('BUF_1_FORM_' + buf1_token['word'])
                buf1word = 1
            if FeatureExtractor._check_informative(buf1_token['tag'], True):
                result.append('BUF_1_POS_' + buf1_token['tag'])
                buf1pos = 1
      
        if stk0pos == 1 and buf0pos == 1:
            result.append('STK_0_BUF_0_POS_' + stk0_token['tag'] + '_' + buf0_token['tag'])
        if stk0word == 1 and buf0word == 1:
            result.append('STK_0_BUF_0_FORM_' + stk0_token['word'] + '_' + buf0_token['word'])
        if FeatureExtractor._check_informative(stk0_token['ctag'], True) and FeatureExtractor._check_informative(buf0_token['ctag'], True):
            result.append('STK_0_BUF_0_CPOS_' + stk0_token['ctag'] + '_' + buf0_token['ctag'])

        if stk0pos == 1 and stkldep != '' and stkrdep != '':
            result.append('STK_0_STKL_STKR_POS_' + stk0_token['tag'] + '_' + stkldep + '_' + stkrdep)


        return result 
