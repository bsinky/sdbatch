use std::time::Duration;

pub fn print_elapsed(duration: &Duration) -> String {
    let duration = chrono::Duration::from_std(*duration)
        .unwrap_or(chrono::Duration::zero());
    if duration.num_hours() > 0 {
        let minutes = duration.num_minutes() - duration.num_hours() * 60;
        let seconds = duration.num_seconds() - duration.num_minutes() * 60;
        return format!("{:02}:{:02}:{:02}", duration.num_hours(), minutes, seconds);
    }
    if duration.num_minutes() > 0 {
        let seconds = duration.num_seconds() - duration.num_minutes() * 60;
        return format!("{:02}:{:02}", duration.num_minutes(), seconds);
    }

    let milliseconds = duration.num_milliseconds() - (duration.num_seconds() * 1000);
    return format!("{}.{}s", duration.num_seconds(), milliseconds);
}

#[cfg(test)]
mod test {
    use core::time;

    use super::*;

    #[test]
    fn less_than_a_minute() {
        let duration = time::Duration::new(49, 19 * 10000000);
        assert_eq!(print_elapsed(&duration), "49.190s");
    }

    #[test]
    fn more_than_a_minute() {
        let duration = time::Duration::new(189, 12345);
        assert_eq!(print_elapsed(&duration), "03:09");
    }

    #[test]
    fn more_than_an_hour() {
        let duration = time::Duration::new(60*60*3+60*9+12, 0);
        assert_eq!(print_elapsed(&duration), "03:09:12");
    }
}
