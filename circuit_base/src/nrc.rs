#[macro_export]
macro_rules! new_rc {
    ($pub:vis fn $fname:ident($($arg:ident : $argtype:ty),* $(,)?) -> ($rettype:ty)  $body:expr)=>{
        $pub fn $fname($($arg : $argtype),*)->$rettype {
            $body
        }

        pub fn nrc($($arg : $argtype),* )->CircuitRc{
            Self::$fname($($arg),* ).rc()
        }
    }
}
#[macro_export]
macro_rules! new_rc_unwrap {
    ($pub:vis fn $fname:ident($( $arg:ident : $argtype:ty),* $(,)?) -> Result<$rettype:ty>  { $($body:tt)* }) =>{
        $pub fn $fname($($arg : $argtype),*)->anyhow::Result<$rettype> {
            $($body)*
        }

        pub fn nrc($($arg : $argtype),*)->CircuitRc{
            Self::$fname($($arg),*).unwrap().rc()
        }
        pub fn new($($arg : $argtype),*)-> $rettype{
            Self::$fname($($arg),*).unwrap()
        }
    }
}
